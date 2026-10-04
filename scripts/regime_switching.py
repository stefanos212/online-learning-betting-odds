"""
REGIME-SWITCHING MIXTURE: a Markov chain over STATES OF THE MARKET, not over
which expert is best.

markov_ogd.py already put a Markov transition on the WEIGHT VECTOR and found
nothing: every kernel selected alpha=0 on train, i.e. collapsed to plain OGD.
fixed_share.py is the same idea in its simplest form and also loses. Both move
mass between experts, and once the step is tuned there is nothing left for that
to fix.

This file changes the object the chain runs on. The latent state here is a
property of the MATCH, not of the expert panel:

    b_t  = belief over K regimes           (a distribution, carried forward)
    w^k  = one weight vector per regime    (each under Sleeping Experts)
    p_t  = sum_k b_t(k) * <w^k restricted to awake, p_t>

After the outcome, the belief is updated by the exact forward recursion of a
hidden Markov model -- Bayes on each regime's predictive likelihood, then a
transition b <- (1-tau) b + tau/K -- and each regime's weight vector takes an
OGD step scaled by its own RESPONSIBILITY b_t(k). A regime that the data says
is not active barely moves, which is what keeps the regimes from collapsing
into each other.

WHY THIS IS WORTH TRYING HERE, when the other Markov variant failed.
information_arrival.py measured a real latent structure in this data: stratified
by how much the line actually moved between opening and kickoff, the closing
advantage runs from -0.00044 in the quietest fifth of matches to -0.01158 in the
noisiest -- a 26x gradient. That is a genuine state of the world, already
quantified, and the panel's relative quality plausibly depends on it. Nothing in
the pipeline currently conditions on it.

THE FALSIFIABLE QUESTION, and the reason the regime is left LATENT rather than
fed in. The obvious design would hand the algorithm the measured movement as a
context variable. Instead the regimes are learned from prediction losses alone,
and afterwards the inferred belief is correlated against the measured movement.
So the file answers two separate things:

  1. does regime switching improve log-loss over plain OGD?
  2. does a 2-state chain, given no side information whatsoever, REDISCOVER the
     information-arrival structure that was measured directly?

A negative answer to (1) with a positive answer to (2) would still be worth
reporting: it would say the structure is real but not exploitable by
reweighting, which is exactly the pattern the heterogeneity results keep
producing.

CAUSALITY NOTE. The movement statistic is used only as a DIAGNOSTIC, after the
fact, never inside the prediction. That matters less than it sounds -- opening
and closing prices are both known before kick-off, so conditioning on them would
be legitimate -- but keeping it out of the algorithm is what makes question (2)
meaningful rather than circular.

SELECTION: tau and K are chosen on the chronological train prefix only, the same
protocol as every other hyperparameter here.

Produces:
  - results/regime_switching.csv
  - results/regime_switching_significance.csv
  - results/regime_switching.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    DATA_DIR, OUT_DIR, EPS, M as LOSS_BOUND, COLORS,
    load_full_universe, per_round_expert_loss, project_to_simplex,
    run_ogd_sleeping, evaluate, OGD_STEP_MULTIPLIER,
)
from calibration_analysis import _style_axes
from final_ranking import calibrate_full_history
from significance_test import moving_block_bootstrap_test, paired_diff

TRAIN_FRACTION = 0.75
TAU_GRID = (0.0, 0.01, 0.1)
# beta extended to 1000 after a first pass selected the old top end (30). An
# answer on a grid boundary is evidence the grid is too narrow, never evidence
# of an optimum -- the lesson learning_rate_study.py paid for.
BETA_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
# gamma = 0 is the genuine EDGE OF THE PARAMETER SPACE, not an under-extended
# grid: negative forgetting would mean amplifying the past, which is meaningless
# here. So a selection at 0 needs no extension, unlike one at the top of beta.
GAMMA_GRID = (0.0, 0.001, 0.01, 0.1)
K_GRID = (2, 3)
TAU_PROFILE = 0.01                     # stage-A tau, interior to TAU_GRID
MIN_PAIRS = 3          # a match needs this many opening/closing pairs to be scored


# ------------------------------------------------- the diagnostic covariate --

def movement_per_match(match_ids):
    """Mean total-variation distance between each bookmaker's opening and
    closing probability vector, over the pairs available for that match.

    Own small copy rather than an import from information_arrival.py, following
    this project's convention for cross-extension dependencies -- that file
    imports half the pipeline, and this one needs fifteen lines of it.
    """
    long = pd.read_csv(os.path.join(DATA_DIR, "odds_long.csv"),
                       usecols=["MatchID", "Bookmaker", "pH", "pD", "pA"])
    piv = long.pivot(index="MatchID", columns="Bookmaker", values=["pH", "pD", "pA"])
    books = sorted(long["Bookmaker"].unique())
    pairs = [(b, b + "C") for b in books if not b.endswith("C") and b + "C" in books]
    tv_sum = pd.Series(0.0, index=piv.index)
    tv_cnt = pd.Series(0, index=piv.index)
    for op, cl in pairs:
        d = sum((piv[(k, cl)] - piv[(k, op)]).abs() for k in ["pH", "pD", "pA"]) / 2.0
        ok = d.notna()
        tv_sum[ok] += d[ok]
        tv_cnt[ok] += 1
    mov = (tv_sum / tv_cnt).where(tv_cnt >= MIN_PAIRS)
    return mov.reindex(match_ids).to_numpy(), len(pairs)


def most_variable_regime(B, mov, ok, n_q=5):
    """Pick the regime whose belief actually varies, and bin the measured
    movement by its quintiles.

    Returns (regime index, list of per-quintile mean movement), or (index, None)
    when no belief varies enough to stratify on. That case is not a failure: a
    chain that concentrates on one regime has a constant belief by construction,
    and the honest report is that there is nothing to correlate.
    """
    sds = [np.std(B[ok, k]) for k in range(B.shape[1])]
    k = int(np.argmax(sds))
    if sds[k] <= 1e-9:
        return k, None
    try:
        q = pd.qcut(B[ok, k], n_q, labels=False, duplicates="drop")
    except ValueError:
        return k, None
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).any():
        return k, None
    n_bins = int(np.nanmax(q)) + 1
    if n_bins < 2:
        return k, None
    return k, [float(mov[ok][q == i].mean()) for i in range(n_bins)]


def chronological_match_ids():
    """MatchIDs in exactly load_full_universe()'s order. Replicated here and
    verified against the dates that loader returns, because a silent
    misalignment would corrupt the diagnostic without raising anything."""
    long = pd.read_csv(os.path.join(DATA_DIR, "odds_long.csv"),
                       usecols=["MatchID", "Bookmaker", "Date", "pH"],
                       parse_dates=["Date"])
    piv = long.pivot(index="MatchID", columns="Bookmaker", values="pH")
    meta = long.drop_duplicates("MatchID").set_index("MatchID")[["Date"]]
    keep = piv.index[piv.notna().any(axis=1)]
    meta = meta.loc[keep].sort_values("Date", kind="mergesort")
    return meta.index.to_numpy(), meta["Date"].to_numpy()


# ------------------------------------------------------------- algorithm ----

def run_regime_ogd(loss, P, y, awake, N, K, tau, beta=1.0, gamma=0.0,
                   clock=True, init_spread=0.30, seed=0):
    """Sleeping-experts OGD replicated across K latent regimes, tied together by
    a hidden Markov chain over the regimes.

    K=1 reduces EXACTLY to run_ogd_sleeping for ANY setting of the knobs: the
    belief is identically 1, the responsibility scaling is a no-op, the
    transition leaves a one-element distribution alone, and the effective clock
    n_1 equals t. main() asserts this.

    THE KNOBS, and why they exist. The first version of this file had only tau,
    and it failed in an instructive way: the mean belief sat at exactly 0.500
    for both regimes and the correlation with measured line movement was zero to
    four decimals. The regimes never DIFFERENTIATED. They receive the same loss
    vector, so from a near-symmetric start they follow near-identical
    trajectories, their likelihoods stay equal, and the belief never moves. The
    result was simply OGD with its step cut by a factor K -- which is exactly
    what the numbers showed. Three knobs address that directly:

      beta  -- sharpness of the belief update. The likelihood enters as
               lik^beta, so beta > 1 makes the belief react to small
               per-round differences instead of averaging them away. This is
               what lets two regimes compete for rounds and specialise.
      gamma -- forgetting. The belief is a DISCOUNTED sum of log-likelihoods,
               so it tracks which regime is better *now* rather than which has
               been better since 2016. With gamma = 0 and beta = 1 the update
               is exact Bayes, i.e. the original behaviour.
      clock -- each regime's step decays on its own accumulated responsibility
               n_k = sum_s b_s(k) rather than on global t. Without it a regime
               that owns half the rounds still decays as if it had seen all of
               them, so the mixture is systematically under-stepped. With it,
               each regime is OGD on its own effective subsequence.

    Returns (phat, B) with phat the (T,3) mixture prediction and B the (T,K)
    belief trajectory.
    """
    D = np.sqrt(2.0)
    T = loss.shape[0]
    W = np.full((K, N), 1.0 / N)          # one persistent weight vector per regime
    if K > 1 and init_spread > 0:
        rng = np.random.default_rng(seed)
        W = W * (1.0 + init_spread * rng.standard_normal((K, N)))
        W = np.maximum(W, 1e-6)
        W = W / W.sum(axis=1, keepdims=True)
    s = np.zeros(K)                       # discounted log-belief, pre-softmax
    n_eff = np.zeros(K)                   # accumulated responsibility per regime
    phat = np.zeros((T, 3))
    B = np.zeros((T, K))

    for t in range(T):
        a = awake[t]
        if not a.any():
            continue
        # belief for THIS round, from everything strictly before it
        e = np.exp(s - s.max())
        b = e / e.sum()
        if K > 1 and tau > 0.0:
            b = (1.0 - tau) * b + tau / K

        V = np.empty((K, int(a.sum())))
        for k in range(K):
            V[k] = project_to_simplex(W[k, a])
            W[k, a] = V[k]
        pk = V @ P[t, a, :]                   # (K, 3) each regime's own mixture
        phat[t] = b @ pk
        B[t] = b

        # discounted, tempered forward recursion
        lik = np.clip(pk[:, y[t]], EPS, 1.0)
        s = (1.0 - gamma) * s + beta * np.log(lik)
        s -= s.max()                          # keep it bounded; softmax is shift-invariant

        # each regime steps in proportion to its responsibility, on its own clock
        for k in range(K):
            n_k = n_eff[k] if clock else t
            eta = OGD_STEP_MULTIPLIER * D / (LOSS_BOUND * np.sqrt(n_k + 1.0))
            W[k, a] = W[k, a] - eta * b[k] * loss[t, a]
        n_eff += b
    return phat, B


# ------------------------------------------------------------------ main ----

def main():
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    loss = per_round_expert_loss(P, y)
    cut = int(round(T * TRAIN_FRACTION))
    print(f"{T} γύροι, {N} experts, train ως {cut}, ουρά {T - cut}\n")

    def scores(phat):
        ll = -np.log(np.clip(phat[np.arange(T), y], EPS, 1.0))
        return ll[:cut].mean(), ll[cut:].mean(), ll

    # ---- sanity: one regime must BE plain OGD ------------------------------
    phat1, _ = run_regime_ogd(loss, P, y, awake, N, K=1, tau=0.0)
    W_ogd = run_ogd_sleeping(loss, awake, N)
    phat_ogd = np.einsum("tn,tnk->tk", W_ogd, P)
    assert np.allclose(phat1, phat_ogd), "K=1 must reproduce run_ogd_sleeping"
    print("sanity ok: K=1 == run_ogd_sleeping\n")
    tr0, va0, ll_ogd = scores(phat_ogd)
    print(f"απλό OGD: train {tr0:.6f}  ουρά {va0:.6f}\n")

    # ---- two-stage sweep, selection on train only --------------------------
    rows = []

    def run(K, tau, beta, gamma, clock, tag):
        phat, B = run_regime_ogd(loss, P, y, awake, N, K, tau,
                                 beta=beta, gamma=gamma, clock=clock)
        tr, va, _ = scores(phat)
        spread = float(np.abs(B - B.mean(axis=1, keepdims=True)).mean())
        rows.append({"stage": tag, "K": K, "tau": tau, "beta": beta, "gamma": gamma,
                     "clock": clock, "train": tr, "held_out": va,
                     "belief_spread": spread})
        print(f"  K={K} τ={tau:<5g} β={beta:<5g} γ={gamma:<6g} clock={str(clock):5s}"
              f" train {tr:.6f}  ουρά {va:.6f}  (διασπορά πεποίθησης {spread:.3f})")
        return tr

    print(f"-- στάδιο A: β x γ, στο K=2, τ={TAU_PROFILE}, clock=True --")
    print("   (η διασπορά πεποίθησης δείχνει αν τα καθεστώτα διαφοροποιούνται καθόλου)")
    bestA = (np.inf, None, None)
    for beta in BETA_GRID:
        for gamma in GAMMA_GRID:
            tr = run(2, TAU_PROFILE, beta, gamma, True, "A")
            if tr < bestA[0]:
                bestA = (tr, beta, gamma)
    _, beta_star, gamma_star = bestA
    print(f"  -> β={beta_star:g}, γ={gamma_star:g}")

    print(f"\n-- στάδιο B: τ, clock και K, στο β={beta_star:g}, γ={gamma_star:g} --")
    bestB = (np.inf, 2, TAU_PROFILE, True)
    for tau in TAU_GRID:
        for clock in (True, False):
            tr = run(2, tau, beta_star, gamma_star, clock, "B")
            if tr < bestB[0]:
                bestB = (tr, 2, tau, clock)
    _, _, tau_star, clock_star = bestB
    for K in K_GRID:
        if K == 2:
            continue
        tr = run(K, tau_star, beta_star, gamma_star, clock_star, "B")
        if tr < bestB[0]:
            bestB = (tr, K, tau_star, clock_star)
    _, K_star, tau_star, clock_star = bestB

    sweep = pd.DataFrame(rows)
    print(f"\nεπιλογή στο train: K={K_star}, τ={tau_star:g}, β={beta_star:g}, "
          f"γ={gamma_star:g}, clock={clock_star}")
    if beta_star in (BETA_GRID[0], BETA_GRID[-1]):
        print("  ! ΠΡΟΣΟΧΗ: το β κάθεται στην ακμή του grid -- χρειάζεται επέκταση")
    if gamma_star == GAMMA_GRID[-1]:
        print("  ! ΠΡΟΣΟΧΗ: το γ κάθεται στην άνω ακμή του grid")
    if gamma_star == 0.0:
        print("  (γ=0 είναι το όριο του χώρου παραμέτρων, όχι του grid: "
              "αρνητική λήθη δεν ορίζεται)")

    phat_star, B_star = run_regime_ogd(loss, P, y, awake, N, K_star, tau_star,
                                       beta=beta_star, gamma=gamma_star,
                                       clock=clock_star)
    tr, va, ll_star = scores(phat_star)
    verdict_tr = "καλύτερο" if tr < tr0 else "χειρότερο"
    verdict_va = "καλύτερο" if va < va0 else "χειρότερο"
    print(f"  -> train {tr:.6f} ({verdict_tr})  ουρά {va:.6f} ({verdict_va})"
          f"   [OGD: {tr0:.6f} / {va0:.6f}]")

    # ---- did the latent chain rediscover the information-arrival structure? --
    print("\n" + "=" * 68)
    print("Διαγνωστικό: το λανθάνον καθεστώς έναντι της ΜΕΤΡΗΜΕΝΗΣ κίνησης γραμμής")
    print("=" * 68)
    mids, mdates = chronological_match_ids()
    assert len(mids) == T, f"match ids {len(mids)} != T {T}"
    assert (mdates == dates).all(), "chronological order does not match the loader"
    mov, n_pairs = movement_per_match(mids)
    ok = ~np.isnan(mov)
    print(f"ζεύγη opening/closing για την κίνηση: {n_pairs}· "
          f"αγώνες με ορισμένη κίνηση: {int(ok.sum())}")

    diag_rows = []
    for k in range(K_star):
        bk = B_star[ok, k]
        # a regime whose belief never moves has no correlation to report -- this
        # is the normal outcome when the chain concentrates, not an error
        r = (float(np.corrcoef(bk, mov[ok])[0, 1])
             if np.std(bk) > 1e-12 else float("nan"))
        diag_rows.append({"regime": k, "pearson_r_vs_movement": r,
                          "mean_belief": float(B_star[:, k].mean()),
                          "belief_sd": float(np.std(bk))})
        shown = f"{r:+.4f}" if np.isfinite(r) else "—  (σταθερή πεποίθηση)"
        print(f"  καθεστώς {k}: μέση πεποίθηση {B_star[:, k].mean():.3f}, "
              f"sd {np.std(bk):.4f}, συσχέτιση με την κίνηση r={shown}")

    kbest, means = most_variable_regime(B_star, mov, ok)
    if means is None:
        print("\n  Καμία πεποίθηση δεν μεταβάλλεται αρκετά ώστε να στρωματοποιηθεί: "
              "το λανθάνον καθεστώς δεν αντιστοιχεί σε τίποτα μετρήσιμο.")
    else:
        print(f"\n  μέση κίνηση ανά πεμπτημόριο πεποίθησης του καθεστώτος {kbest}:")
        for i, m in enumerate(means):
            print(f"    Q{i+1}: {m:.5f}")

    # ---- full comparison, identical rounds ---------------------------------
    label_star = (f"Regime-OGD (K={K_star}, β={beta_star:g}, γ={gamma_star:g}, "
                  f"τ={tau_star:g})")
    entries = {"OGD": phat_ogd, label_star: phat_star}
    alive = np.ones(T, dtype=bool)
    table, per_round = [], {}
    for label, ph in entries.items():
        p, cal, yy, _ = calibrate_full_history(ph, y, alive)
        raw_m, cal_m = evaluate(p, yy), evaluate(cal, yy)
        per_round[label] = -np.log(np.clip(cal[np.arange(T), yy], EPS, 1.0))
        table.append({"name": label, "raw_log_loss": raw_m["log_loss"],
                      "calibrated_log_loss": cal_m["log_loss"],
                      "raw_brier": raw_m["brier"], "accuracy": cal_m["accuracy"]})
    tbl = pd.DataFrame(table)
    pd.set_option("display.width", 200)
    print(f"\n=== Σύγκριση, ίδιοι {T} γύροι ===\n")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    d, n = paired_diff(per_round[list(entries)[1]], per_round["OGD"])
    r = moving_block_bootstrap_test(d)
    verdict = ("ισοπαλία" if not r["significant_95"]
               else ("Regime-OGD better" if r["mean_diff"] < 0 else "OGD better"))
    print(f"\nΈναντι OGD (αρνητικό = καλύτερο): mean_diff {r['mean_diff']:+.6f}, "
          f"p={r['p_value']:.3f} -> {verdict}")
    sig = pd.DataFrame([{"series": list(entries)[1], "vs": "OGD", "n": n,
                         "verdict": verdict, **r}])

    sp = os.path.join(OUT_DIR, "regime_switching.csv")
    sgp = os.path.join(OUT_DIR, "regime_switching_significance.csv")
    pd.concat([sweep, pd.DataFrame(diag_rows), tbl], ignore_index=True).to_csv(sp, index=False)
    sig.to_csv(sgp, index=False)
    print(f"\nSaved: {sp}\nSaved: {sgp}")
    plot_regimes(sweep, B_star, mov, ok, tau_star, K_star, va0,
                 os.path.join(OUT_DIR, "regime_switching.png"))


def plot_regimes(sweep, B, mov, ok, tau_star, K_star, base_ho, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="#fcfcfb")
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
    palette = [COLORS["Hedge"], COLORS["FTRL"], COLORS["Uniform average"],
               COLORS["muted"]]
    ax1.axhline(base_ho, color=COLORS["OGD"], linewidth=1.6, label="απλό OGD")
    A = sweep[sweep["stage"] == "A"]
    for i, g in enumerate(sorted(A["gamma"].unique())):
        s = A[A["gamma"] == g].sort_values("beta")
        ax1.plot(s["beta"], s["held_out"], marker="o", markersize=6, linewidth=2.0,
                 color=palette[i % len(palette)], label=f"γ={g:g}")
    ax1.set_xscale("log")
    ax1.set_xlabel("οξύτητα πεποίθησης β (λογαριθμικός)", color="#52514e")
    ax1.set_ylabel("log-loss στην ουρά", color="#52514e")
    ax1.set_title("Εναλλαγή καθεστώτων: σάρωση", color="#0b0b0b", fontsize=12, pad=10)
    _style_axes(ax1)
    leg = ax1.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color("#0b0b0b")

    kb, means = most_variable_regime(B, mov, ok)
    if means is None:
        ax2.text(0.5, 0.5, "η πεποίθηση δεν μεταβάλλεται·\nδεν υπάρχει τίποτα "
                           "να στρωματοποιηθεί", ha="center", va="center",
                 transform=ax2.transAxes, color="#52514e", fontsize=11)
        ax2.set_xticks([])
        ax2.set_yticks([])
    else:
        ax2.plot(range(1, len(means) + 1), means, marker="o", markersize=9,
                 linewidth=2.0, color=COLORS["Hedge"])
        ax2.set_xticks(range(1, len(means) + 1))
        ax2.set_ylabel("μέση μετρημένη κίνηση γραμμής", color="#52514e")
    ax2.set_xlabel(f"πεμπτημόριο πεποίθησης για το καθεστώς {kb}", color="#52514e")
    ax2.set_title("Ανακαλύπτει το λανθάνον καθεστώς την άφιξη πληροφορίας;",
                  color="#0b0b0b", fontsize=12, pad=10)
    _style_axes(ax2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
