"""
BAYESIAN (MCMC) KELLY: letting the shrinkage emerge instead of imposing it.

The Kelly criterion assumes the probability is KNOWN. Ours is estimated, and
Kelly is notoriously sensitive to that: overstating p by a little overbets by a
lot, and the log-growth objective punishes overbetting far more than
underbetting. The project's response so far is the standard workaround --
fractional Kelly at 1/4 -- which works but is arbitrary. Nothing in the data
chose the 1/4.

The principled alternative is to treat theta, the true outcome distribution, as
uncertain and maximise EXPECTED log growth under its posterior:

    f* = argmax_f  E_{theta ~ posterior} [ theta_k log(1 + f(o_k - 1))
                                          + (1 - theta_k) log(1 - f) ]

Because log(1-f) -> -infinity as f -> 1, posterior mass on low theta_k punishes
large stakes automatically. The shrinkage is then a CONSEQUENCE of uncertainty
rather than a constant someone picked, and its size adapts per match: confident
rounds get bet harder, contested ones softer.

THE MODEL. At a betting round the awake experts supply forecasts p_1..p_K with
learned weights w_1..w_K. We treat their disagreement as evidence about theta:

    theta   ~  Dirichlet(1, 1, 1)                     (flat prior)
    p_i     ~  Dirichlet(kappa * theta)                for each awake expert i

with each expert's contribution weighted by w_i. kappa controls how tightly
experts are assumed to cluster around the truth: large kappa means a small
spread among experts implies a sharp posterior. The likelihood is Dirichlet in
its PARAMETER, which is not conjugate, so the posterior has no closed form and
is sampled by random-walk Metropolis-Hastings on the simplex.

kappa is selected on the chronological train prefix, the same protocol as every
other hyperparameter here.

WHAT IS COMPARED, all on identical rounds and identical probabilities:
    plug-in full     f* from the point estimate, no shrinkage
    plug-in 1/4      what the pipeline currently ships
    MCMC full        posterior-expected objective, no shrinkage imposed
    MCMC 1/4         both, to separate the two effects

SIGN WARNING: this file scores log-GROWTH per bet, where HIGHER is better --
the opposite of every log-loss table in the project.

Produces:
  - results/mcmc_kelly.csv
  - results/mcmc_kelly.png
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import OUT_DIR, EPS, COLORS, load_match_order, load_universe_from, split_by_market_phase
from calibration_analysis import _style_axes
from significance_test import moving_block_bootstrap_test
from betting import (
    STARTING_BANKROLL, fit_leave_out, kelly_fraction,
)
from sleeping_experts import DATA_DIR

TRAIN_FRACTION = 0.75
KAPPA_GRID = (20.0, 60.0, 200.0)
N_SAMPLES = 64           # parallel chains = posterior draws per round
N_BURN = 25              # iterations each chain runs before its state is kept
PROPOSAL_CONC = 900.0    # random-walk tightness on the simplex
TARGETS = ("WHC", "PSC", "BWC", "B365C")
RNG_SEED = 12345
KAPPA_SELECT_STRIDE = 3
"""Every k-th eligible train round is used to pick kappa. The sampler is the
expensive step (a 40-iteration, 64-chain MH run PER CANDIDATE BET), so scoring
every round for all of KAPPA_GRID is the dominant cost. Subsampling trades a
little selection precision for a large constant-factor speedup; kappa is a
coarse smoothing choice, not a quantity that needs per-round accuracy."""


# ------------------------------------------------------------- posterior --

_LANCZOS = np.array([
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7])


def lgamma(x):
    """Vectorised log-Gamma for x > 0 (Lanczos, g=7).

    scipy is not a dependency of this project, and math.lgamma inside the
    sampler's inner loop would dominate the runtime, so the approximation is
    carried here. Accuracy is verified against math.lgamma on import.
    """
    z = np.asarray(x, dtype=float) - 1.0
    a = np.full(z.shape, _LANCZOS[0])
    for i in range(1, 9):
        a = a + _LANCZOS[i] / (z + i)
    t = z + 7.5
    return 0.5 * math.log(2 * math.pi) + (z + 0.5) * np.log(t) - t + np.log(a)


assert np.allclose([lgamma(v) for v in (0.3, 1.0, 2.5, 17.0, 900.0)],
                   [math.lgamma(v) for v in (0.3, 1.0, 2.5, 17.0, 900.0)],
                   rtol=1e-10), "lgamma approximation is off"


def _dir_logpdf(x, alpha):
    """log Dir(x; alpha) with broadcasting over leading axes."""
    return (lgamma(alpha.sum(-1)) - lgamma(alpha).sum(-1)
            + ((alpha - 1.0) * np.log(x)).sum(-1))


def sample_posterior(p_experts, weights, kappa, rng,
                     n=N_SAMPLES, burn=N_BURN, conc=PROPOSAL_CONC):
    """Metropolis-Hastings for theta given the expert forecasts.

    Runs `n` chains IN PARALLEL for `burn` iterations and keeps their final
    states, rather than one chain for n+burn iterations. Statistically the
    draws are independent (which a single chain's are not); computationally it
    turns a scalar Python loop into `burn` vectorised numpy steps, which is
    what makes the per-round cost affordable across thousands of bets.

    The proposal is Dirichlet centred on the current state and therefore
    asymmetric, so the acceptance ratio carries the proposal densities in both
    directions.
    """
    P = np.clip(p_experts, EPS, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    w = weights / weights.sum()

    def loglik(th):
        """(S,3) states -> (S,) weighted log-likelihood of the expert panel."""
        a = kappa * np.clip(th, 1e-9, None)
        lp = _dir_logpdf(P[None, :, :], a[:, None, :])     # (S, K)
        return lp @ w

    start = np.clip(w @ P, 1e-6, None)
    start = start / start.sum()
    cur = rng.dirichlet(conc * start, size=n)
    cur = np.clip(cur, 1e-9, None)
    cur /= cur.sum(axis=1, keepdims=True)
    cur_ll = loglik(cur)

    for _ in range(burn):
        # per-chain Dirichlet proposal centred on that chain's current state
        prop = _dirichlet_rows(rng, conc * cur)
        prop = np.clip(prop, 1e-9, None)
        prop /= prop.sum(axis=1, keepdims=True)
        prop_ll = loglik(prop)
        fwd = _dir_logpdf(prop, conc * cur)
        bwd = _dir_logpdf(cur, conc * prop)
        acc = np.log(rng.random(n)) < (prop_ll - cur_ll) + (bwd - fwd)
        cur[acc] = prop[acc]
        cur_ll[acc] = prop_ll[acc]
    return cur


def _dirichlet_rows(rng, alpha):
    """Row-wise Dirichlet draws for an (S,3) matrix of concentrations, via the
    gamma construction -- numpy's dirichlet takes a single parameter vector."""
    g = rng.gamma(alpha)
    g = np.clip(g, 1e-12, None)
    return g / g.sum(axis=1, keepdims=True)


def posterior_kelly(theta_draws, k, odds_k, mult, grid=None):
    """Stake maximising posterior-expected log growth on outcome k.

    The objective is concave in f, so a coarse grid followed by the usual
    fractional multiplier is enough; solving exactly would not change the
    ranking of the variants being compared.
    """
    if grid is None:
        grid = np.linspace(0.0, 0.95, 96)
    th = theta_draws[:, k][:, None]
    b = odds_k - 1.0
    gains = np.log1p(grid[None, :] * b)
    losses = np.log(np.clip(1.0 - grid[None, :], 1e-12, None))
    obj = (th * gains + (1.0 - th) * losses).mean(axis=0)
    return float(mult * grid[int(np.argmax(obj))])


# ----------------------------------------------------------- simulation --

def raw_mcmc_stakes(p_hat, odds, y, mask, P_sub, W_sub, kappa, rng, stride=1):
    """The expensive step, run ONCE: the full-Kelly (mult=1) posterior stake
    for every eligible round. `apply_mult` below turns this into growth curves
    for any multiplier without resampling -- the posterior draws don't depend
    on the multiplier, only the final stake = mult * argmax does.

    `stride` evaluates only every stride-th eligible round and leaves the rest
    unplaced; used to speed up kappa SELECTION, never the reported result.
    """
    T = len(y)
    stakes = np.full(T, np.nan)
    seen = 0
    for t in range(T):
        if not mask[t]:
            continue
        ev = p_hat[t] * odds[t] - 1.0
        k = int(np.argmax(ev))
        if ev[k] <= 0:
            continue
        seen += 1
        if stride > 1 and (seen % stride) != 0:
            continue
        aw = W_sub[t] > 0
        if not aw.any():
            continue
        draws = sample_posterior(P_sub[t, aw, :], W_sub[t, aw], kappa, rng)
        f = posterior_kelly(draws, k, odds[t, k], 1.0)
        if f > 0:
            stakes[t] = f
    return stakes


def apply_mult(stakes, mult, odds, y, argmax_k):
    """Growth curve for a given multiplier, from precomputed mult=1 stakes."""
    T = len(stakes)
    growth = np.ones(T)
    placed = ~np.isnan(stakes)
    out_stakes = np.full(T, np.nan)
    for t in np.flatnonzero(placed):
        f = mult * stakes[t]
        k = argmax_k[t]
        out_stakes[t] = f
        growth[t] = 1.0 + f * (odds[t, k] - 1.0) if y[t] == k else 1.0 - f
    return growth, placed, out_stakes


def argmax_outcomes(p_hat, odds, mask):
    """Which outcome has positive EV each round (or -1), point-estimate only --
    shared between plug-in and MCMC so both agree on WHICH bet is being sized,
    and only the size differs."""
    T = len(mask)
    k = np.full(T, -1, dtype=int)
    for t in range(T):
        if not mask[t]:
            continue
        ev = p_hat[t] * odds[t] - 1.0
        kk = int(np.argmax(ev))
        if ev[kk] > 0:
            k[t] = kk
    return k


def simulate_plugin(p_hat, odds, y, mask, mult, argmax_k):
    T = len(y)
    growth = np.ones(T)
    placed = np.zeros(T, dtype=bool)
    stakes = np.full(T, np.nan)
    for t in np.flatnonzero(mask & (argmax_k >= 0)):
        k = argmax_k[t]
        f = mult * kelly_fraction(p_hat[t, k], odds[t, k])
        if f <= 0:
            continue
        placed[t] = True
        stakes[t] = f
        growth[t] = 1.0 + f * (odds[t, k] - 1.0) if y[t] == k else 1.0 - f
    return growth, placed, stakes


def score(growth, placed, mask, label, extra):
    bank = STARTING_BANKROLL * np.cumprod(growth[mask])
    lg = np.log(growth[placed])
    n = int(placed.sum())
    res = (moving_block_bootstrap_test(lg) if n >= 30 else
           {"mean_diff": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "p_value": np.nan, "significant_95": False})
    return {**extra, "variant": label, "n_bets": n,
            "final_bankroll": bank[-1] if len(bank) else np.nan,
            "mean_log_growth_per_bet": res["mean_diff"],
            "ci_low": res["ci_low"], "ci_high": res["ci_high"],
            "p_value": res["p_value"], "significant_95": res["significant_95"]}


def main():
    rng = np.random.default_rng(RNG_SEED)
    match_ids = load_match_order()
    panel, P, awake, y, dates, seasons = load_universe_from(
        os.path.join(DATA_DIR, "odds_long.csv"))
    N = len(panel)
    open_idx, close_idx = split_by_market_phase(panel)
    print(f"{P.shape[0]} γύροι, {N} experts, στόχοι: {list(TARGETS)}\n")

    rows, sig_rows, stake_store = [], [], {}
    for book in TARGETS:
        is_closing = book.endswith("C")
        idx = close_idx if is_closing else open_idx
        # Own leave-one-out fit, rather than value_betting.fit_leave_out, so the
        # LEARNED weights stay available: the posterior conditions on how much
        # the mixture trusts each expert, not merely on who is awake.
        P_sub, W_sub, phat_cal, odds, mask, y_sub = fit_with_weights(
            panel, P, awake, y, idx, book, match_ids)

        n_tr = int(round(len(y_sub) * TRAIN_FRACTION))
        tr = np.zeros(len(y_sub), bool); tr[:n_tr] = True
        print(f"--- {book} ({'closing' if is_closing else 'opening'}), "
              f"{len(y_sub)} γύροι, {P_sub.shape[1]} experts στο panel ---")

        argmax_k = argmax_outcomes(phat_cal, odds, mask)
        growth_store = {}

        # plug-in baselines
        for mult, lab in ((1.0, "plug-in, πλήρες"), (0.25, "plug-in, 1/4")):
            g, pl, st = simulate_plugin(phat_cal, odds, y_sub, mask, mult, argmax_k)
            rows.append(score(g, pl, mask, lab, {"bookmaker": book}))
            growth_store[lab] = (g, pl)
            print(f"  {lab:18s} {rows[-1]['n_bets']:6d} στοιχ.  "
                  f"κεφ. {rows[-1]['final_bankroll']:.3g}  "
                  f"g {rows[-1]['mean_log_growth_per_bet']:+.5f}")

        # kappa selected on the train prefix only, subsampled for speed --
        # the sampler is the expensive step and kappa is a coarse choice
        best_k, best_g = None, -np.inf
        for kap in KAPPA_GRID:
            st = raw_mcmc_stakes(phat_cal, odds, y_sub, mask & tr, P_sub, W_sub,
                                 kap, rng, stride=KAPPA_SELECT_STRIDE)
            g, pl, _ = apply_mult(st, 1.0, odds, y_sub, argmax_k)
            v = np.log(g[pl]).mean() if pl.sum() >= 30 else -np.inf
            print(f"    κ={kap:g}: {int(pl.sum())} δείγματα (stride "
                  f"{KAPPA_SELECT_STRIDE}), μέσο log-growth {v:+.5f}")
            if v > best_g:
                best_g, best_k = v, kap
        print(f"  κ επιλεγμένο στο train: {best_k:g}")

        # ONE sampler pass at mult=1 on ALL eligible rounds; the 1/4 variant
        # reuses these draws instead of resampling from scratch
        raw_st = raw_mcmc_stakes(phat_cal, odds, y_sub, mask, P_sub, W_sub,
                                 best_k, rng, stride=1)
        for mult, lab in ((1.0, "MCMC, πλήρες"), (0.25, "MCMC, 1/4")):
            g, pl, st = apply_mult(raw_st, mult, odds, y_sub, argmax_k)
            r = score(g, pl, mask, lab, {"bookmaker": book})
            r["kappa"] = best_k
            r["mean_stake"] = float(np.nanmean(st))
            rows.append(r)
            growth_store[lab] = (g, pl)
            if mult == 1.0:
                stake_store[book] = st[pl]
            print(f"  {lab:18s} {r['n_bets']:6d} στοιχ.  "
                  f"κεφ. {r['final_bankroll']:.3g}  "
                  f"g {r['mean_log_growth_per_bet']:+.5f}  "
                  f"μέσο πόντισμα {r['mean_stake']:.4f}")

        # paired test: MCMC vs plug-in, SAME fractional multiplier, restricted
        # to rounds where BOTH variants actually placed a bet (each may decline
        # different rounds, so the intersection is the only fair comparison)
        for mult_lab in ("πλήρες", "1/4"):
            g_p, pl_p = growth_store[f"plug-in, {mult_lab}"]
            g_m, pl_m = growth_store[f"MCMC, {mult_lab}"]
            both = pl_p & pl_m
            n_both = int(both.sum())
            if n_both >= 30:
                d = np.log(g_m[both]) - np.log(g_p[both])
                res = moving_block_bootstrap_test(d)
                verdict = ("ισοπαλία" if not res["significant_95"]
                           else ("MCMC" if res["mean_diff"] > 0 else "plug-in"))
                sig_rows.append({"bookmaker": book, "mult": mult_lab, "n": n_both,
                                 "mean_diff": res["mean_diff"],
                                 "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                                 "p_value": res["p_value"], "verdict": verdict})
                print(f"  [{mult_lab}] MCMC έναντι plug-in, κοινοί {n_both} "
                      f"γύροι: diff {res['mean_diff']:+.5f}  "
                      f"p={res['p_value']:.4f}  -> {verdict}")
        print()

    sig_table = pd.DataFrame(sig_rows)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=== Σύνοψη ===\n")
    print(table[["bookmaker", "variant", "n_bets", "final_bankroll",
                 "mean_log_growth_per_bet", "p_value", "significant_95"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n=== MCMC έναντι plug-in, ζευγαρωτά στους κοινούς γύρους ===")
    print("    (θετική διαφορά = μεγαλύτερος ρυθμός ανάπτυξης του MCMC)\n")
    print(sig_table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    out = os.path.join(OUT_DIR, "mcmc_kelly.csv")
    sgp = os.path.join(OUT_DIR, "mcmc_kelly_significance.csv")
    table.to_csv(out, index=False)
    sig_table.to_csv(sgp, index=False)
    print(f"\nSaved: {out}\nSaved: {sgp}")
    plot_stakes(stake_store, os.path.join(OUT_DIR, "mcmc_kelly.png"))


def fit_with_weights(panel, P, awake, y, indices, book, match_ids):
    """Leave-one-out OGD fit that also returns the per-round weights.

    Reproduces value_betting.fit_leave_out's round reduction exactly (keep the
    rounds where at least one remaining expert is awake AND the target quoted
    usable odds), so every returned array is aligned to the same rounds.
    """
    from betting import load_decimal_odds
    from sleeping_experts import per_round_expert_loss, run_ogd_sleeping
    from final_ranking import calibrate_full_history

    col = panel.index(book)
    keep_cols = [k for k in indices if k != col]
    Pk, awk = P[:, keep_cols, :], awake[:, keep_cols]
    odds_full = load_decimal_odds(book, match_ids)
    target_ok = awake[:, col] & ~np.isnan(odds_full[:, 0])

    keep = awk.any(axis=1) & target_ok
    P_sub, aw_sub, y_sub = Pk[keep], awk[keep], y[keep]
    odds = odds_full[keep]

    W = run_ogd_sleeping(per_round_expert_loss(P_sub, y_sub), aw_sub, len(keep_cols))
    phat = np.einsum("tn,tnk->tk", W, P_sub)
    all_true = np.ones(len(y_sub), dtype=bool)
    _, phat_cal, _, _ = calibrate_full_history(phat, y_sub, all_true)

    mask = np.ones(len(y_sub), dtype=bool)
    return P_sub, W, phat_cal, odds, mask, y_sub


def plot_stakes(stake_store, out_path):
    if not stake_store:
        return
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    cols = [COLORS["OGD"], COLORS["Hedge"], COLORS["FTRL"], COLORS["Uniform average"]]
    for (book, st), c in zip(stake_store.items(), cols):
        st = st[np.isfinite(st)]
        if len(st):
            ax.hist(st, bins=40, histtype="step", linewidth=1.8, color=c,
                    label=f"{book} (μέσο {st.mean():.3f})", density=True)
    ax.axvline(0.25, color="#898781", linestyle="--", linewidth=1.4,
               label="σταθερό 1/4")
    ax.set_xlabel("κλάσμα ποντίσματος από τον posterior", color="#52514e")
    ax.set_ylabel("πυκνότητα", color="#52514e")
    ax.set_title("Το μέγεθος ποντίσματος που προκύπτει από την αβεβαιότητα",
                 color="#0b0b0b", fontsize=12, pad=12)
    _style_axes(ax)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color("#0b0b0b")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
