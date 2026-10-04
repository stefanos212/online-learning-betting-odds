"""
OCO ON THE KELLY MULTIPLIER (not on the stake).

`value_betting_online_kelly.py` already framed bet sizing as a third OCO stage
and lost to the closed-form formula in 17 of 20 comparisons. This file argues
the loss was a consequence of two specific design choices rather than of the
OCO framing, and fixes both.

WHAT THE EARLIER VERSION DID, AND WHY IT COULD NOT WIN
------------------------------------------------------
It learned a single scalar `f` -- one stake fraction, held constant across all
bets, updated by projected OGD from realized win/loss outcomes. Two problems:

1. **The comparator class is wrong.** A regret bound for that learner is a
   promise about the best CONSTANT f in hindsight. The closed-form Kelly rule
   is not in that class at all: it produces a DIFFERENT f every round, computed
   from p_hat and the odds, both of which are known BEFORE the bet is placed.
   Driving regret to zero against the best constant stake still does not reach a
   round-varying rule. The learner also discards p_hat entirely -- the very
   quantity the two preceding OCO stages exist to produce -- and re-derives its
   stake from one bit per bet.

2. **The algorithm is wrong for the loss class.** The negative log-wealth loss
   is not merely convex, it is EXP-CONCAVE, and projected OGD on a worst-case
   1/sqrt(t) schedule cannot exploit that. Online Newton Step does, at
   O(log T) instead of O(sqrt(T)).

WHAT THIS FILE LEARNS INSTEAD
-----------------------------
The MULTIPLIER, not the stake:

    f_t  =  lambda_t * f*_t ,     f*_t = kelly_fraction(p_hat_t, O_t)

so the round-specific information stays in f*_t and only the shrinkage is
learned. Three things follow.

  * The comparator "best constant lambda in hindsight" is now a meaningful
    object, and it is EXACTLY the quantity the project currently hardcodes:
    KELLY_MULT = 1/4. So this asks "is 1/4 right?" rather than "can something
    beat 1/4?", and a null answer is still informative.
  * Convexity survives. With w the win indicator,
        loss(lambda) = -log(1 - lambda f* + lambda f* O w)
    is -log of an AFFINE function of lambda, hence convex, with gradient
        win : -f*(O-1) / growth        lose : +f* / growth
    (the stake-learner's gradient times f*, by the chain rule).
  * Exp-concavity survives too: exp(-alpha * loss) = (affine)^alpha is concave
    for alpha <= 1, so the loss is 1-exp-concave on any domain where growth
    stays bounded away from 0. That is what admits Online Newton Step.

ONS IN ONE DIMENSION is unusually simple, which is worth stating because it
looks too short to be the real algorithm:
    A_t      = A_{t-1} + g_t^2
    lambda   = clip( lambda - (1/gamma) * g_t / A_t ,  0, LAM_MAX )
The textbook version projects in the A_t-norm; in 1-D that norm is a positive
rescaling of the Euclidean one, so the projection is ordinary clipping.

INITIALISATION is lambda_0 = 1.0, i.e. FULL Kelly, and this is a deliberate
choice rather than a neutral one. If p_hat were exactly correct, full Kelly
would be optimal; shrinkage below 1 is a response to estimation error. Starting
at 1.0 therefore starts the learner at the theoretically right answer under its
own model and makes it discover the shrinkage from data. Starting instead at
0.25 would hand it the project's existing answer, and starting at 0 would
repeat the earlier version's cold start, which is what its post-mortem blamed.

THE OPTIMISATION STAGE lives entirely inside this file's own layer -- the
mixture weights and the calibration temperature are taken as given and are
never re-tuned here. Selection is on a chronological prefix of the BETS
(not of the matches), pooled across targets, in the project's primary
configuration; the remaining bets are held out and never used to choose.
`_extend_if_endpoint` enforces the project's rule that a winner on a grid
boundary means the grid is too narrow.

Produces:
  - results/oco_kelly.csv
  - results/oco_kelly_significance.csv
  - results/oco_kelly.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sleeping_experts import OUT_DIR, DATA_DIR, COLORS, load_match_order, load_universe_from, split_by_market_phase
from significance_test import moving_block_bootstrap_test
from betting import (
    STARTING_BANKROLL, kelly_fraction, select_bookmakers, simulate_kelly_betting,
)
from value_betting_online_kelly import fit_closing_on_closing, simulate_online_kelly

TRAIN_FRACTION = 0.75
LAM_0 = 1.0                      # start at full Kelly -- see module docstring
LAM_MAX_GRID = (1.0, 1.5)
LAM_0_GRID = (0.25, 0.5, 1.0)
LAM_SAFETY = 0.95        # worst-case loss must leave at least 1 - 0.95 of the bankroll
RATE_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
ALGOS = ("ons", "ogd")
MAX_EXTENSIONS = 4
GROWTH_FLOOR = 1e-3              # a stake this close to total loss never occurs; guards log(0)


# ------------------------------------------------------- the bet sequence --

def bet_sequence(p_hat, odds, y, mask):
    """The bets that WILL be placed, independent of any stake multiplier.

    Outcome selection (highest positive-EV outcome, if any) uses only p_hat and
    the odds, so it is identical for every method compared here -- which is what
    makes a paired per-bet test valid. Returns, for each bet in chronological
    order: the round index, the full-Kelly fraction f*, the odds taken, and
    whether it won.
    """
    T = len(y)
    idx, fstar, took, won = [], [], [], []
    for t in range(T):
        if not mask[t]:
            continue
        ev = p_hat[t] * odds[t] - 1.0
        k = int(np.argmax(ev))
        if ev[k] <= 0:
            continue
        f = float(kelly_fraction(p_hat[t, k], odds[t, k]))
        if f <= 0:
            continue                       # matches simulate_kelly_betting's skip
        idx.append(t)
        fstar.append(f)
        took.append(float(odds[t, k]))
        won.append(bool(y[t] == k))
    return (np.array(idx, dtype=int), np.array(fstar), np.array(took),
            np.array(won, dtype=bool))


def _growth(lam, fstar, O, win):
    f = lam * fstar
    return (1.0 - f + f * O) if win else (1.0 - f)


def _grad(lam, fstar, O, win, g):
    """d/dlambda of -log(growth). Chain rule on the stake-learner's gradient."""
    g = max(g, GROWTH_FLOOR)
    return (-fstar * (O - 1.0) / g) if win else (fstar / g)


# ------------------------------------------------------------ simulators --

def simulate_fixed(seq, lam, T):
    """Constant multiplier. Asserted in main() to reproduce
    value_betting.simulate_kelly_betting exactly at the shipped 1/4."""
    idx, fstar, O, win = seq
    growth = np.ones(T)
    placed = np.zeros(T, dtype=bool)
    for j, t in enumerate(idx):
        growth[t] = _growth(lam, fstar[j], O[j], win[j])
        placed[t] = True
    return growth, placed, np.full(T, lam)


def simulate_learned(seq, T, algo="ons", rate=1.0, lam_0=LAM_0, lam_max=1.0):
    """Learn the multiplier online.

    `rate` multiplies the theory-derived constant (1/gamma for ONS, the step for
    OGD). It exists because every worst-case constant in this project has turned
    out to be badly scaled for this data; it is selected on train like any other
    hyperparameter, never on the held-out bets.
    """
    idx, fstar, O, win = seq
    n = len(idx)
    growth = np.ones(T)
    placed = np.zeros(T, dtype=bool)
    lam_trace = np.full(T, np.nan)
    if n == 0:
        return growth, placed, lam_trace

    # honest gradient bound over the whole domain, computed from the data:
    #   win branch  |g| = f*(O-1)/(1+lam f*(O-1)) <= f*(O-1) = EV   (at lam=0)
    #   lose branch |g| = f*/(1-lam f*)                             (at lam=lam_max)
    fmax = float(fstar.max())
    if lam_max * fmax >= 1.0 - GROWTH_FLOOR:
        raise ValueError(f"lam_max={lam_max} x max f*={fmax:.3f} would allow total ruin")
    G = max(float((fstar * (O - 1.0)).max()), fmax / (1.0 - lam_max * fmax))
    D = lam_max
    gamma = 0.5 * min(1.0 / (4.0 * G * D), 1.0)     # alpha = 1 (exp-concavity)

    lam = float(np.clip(lam_0, 0.0, lam_max))
    A = 1.0 / (gamma ** 2 * D ** 2)                 # ONS's standard epsilon
    for j, t in enumerate(idx):
        lam_trace[t] = lam
        g_t = _growth(lam, fstar[j], O[j], win[j])
        growth[t] = g_t
        placed[t] = True
        grad = _grad(lam, fstar[j], O[j], win[j], g_t)
        if algo == "ons":
            A += grad * grad
            lam = lam - rate * (1.0 / gamma) * grad / A
        elif algo == "ogd":
            eta = rate * D / (G * np.sqrt(j + 1))
            lam = lam - eta * grad
        else:
            raise ValueError(algo)
        lam = float(np.clip(lam, 0.0, lam_max))
    return growth, placed, lam_trace


# ------------------------------------------------------------ evaluation --

def log_growth(seq, growth):
    return np.log(np.maximum(growth[seq[0]], GROWTH_FLOOR))


def _extend_if_endpoint(values, evaluate_one, scores, max_ext=MAX_EXTENSIONS):
    """Widen a geometric grid while its best point sits on an edge (see
    learning_rate_study.py and CLAUDE.md -- an endpoint is not an optimum).
    NOTE the sign: these are log-GROWTH scores, so HIGHER is better."""
    for _ in range(max_ext):
        i = int(np.argmax(scores))
        if 0 < i < len(values) - 1:
            return False
        step = values[1] / values[0] if i == 0 else values[-1] / values[-2]
        x = values[0] / step if i == 0 else values[-1] * step
        print(f"      ! ακμή του grid: επεκτείνω σε {x:g}")
        s = evaluate_one(x)
        if i == 0:
            values.insert(0, x)
            scores.insert(0, s)
        else:
            values.append(x)
            scores.append(s)
    return True


def main():
    books = select_bookmakers()
    targets_by_phase = {
        "closing": [b for b in books if b.endswith("C")],
        "opening": [b for b in books if not b.endswith("C")],
    }
    targets = targets_by_phase["closing"]
    print(f"Closing-side στόχοι: {targets_by_phase['closing']}")
    print(f"Opening-side στόχοι: {targets_by_phase['opening']}\n")
    match_ids = load_match_order()

    # ---- fit once per (normalisation, target); the sweep then costs nothing --
    # fit_closing_on_closing is phase-agnostic -- it trains on whatever column
    # subset it is handed -- so passing open_idx gives the opening analogue, and
    # with it the causality constraint the opening case requires: the ENTIRE
    # closing side leaves the training panel, not merely the target, because no
    # closing price exists yet at the moment an opening price is taken.
    print("Προσαρμογή μειγμάτων (μία φορά ανά normalization x στόχο)...")
    cache = {}
    for norm_label, path in [("Basic", os.path.join(DATA_DIR, "odds_long.csv")),
                             ("Shin", os.path.join(DATA_DIR, "odds_long_shin.csv"))]:
        panel, P, awake, y, dates, seasons = load_universe_from(path)
        open_idx, close_idx = split_by_market_phase(panel)
        for phase, idx in (("closing", close_idx), ("opening", open_idx)):
            for book in targets_by_phase[phase]:
                phat_raw, phat_cal, odds, book_mask, y_sub = fit_closing_on_closing(
                    panel, P, awake, y, book, idx, match_ids)
                for stage, p_hat in (("raw", phat_raw), ("calibrated", phat_cal)):
                    seq = bet_sequence(p_hat, odds, y_sub, book_mask)
                    cache[(norm_label, book, stage)] = {
                        "seq": seq, "T": len(y_sub), "p_hat": p_hat, "odds": odds,
                        "y": y_sub, "mask": book_mask, "phase": phase}
                print(f"  {norm_label:6s} {phase:7s} {book:6s} bets: "
                      f"raw {len(cache[(norm_label, book, 'raw')]['seq'][0])}, "
                      f"cal {len(cache[(norm_label, book, 'calibrated')]['seq'][0])}")

    # ---- the admissible lambda domain is a property of the DATA -------------
    # lambda multiplies f*, so lambda * max(f*) < 1 is a hard solvency
    # constraint, not a tuning choice: above it a single loss can take the whole
    # bankroll and the loss function stops being bounded. The ceiling is
    # therefore derived from the observed f*, and any grid value above it is
    # dropped rather than silently clipped.
    # Computed PER PHASE, not once over the pooled cache: the constraint is a
    # property of the bets actually placed against that phase, and keeping them
    # separate is also what leaves the closing-side numbers identical to the
    # version of this study that had no opening side.
    lam_max_grid_by_phase = {}
    for phase in targets_by_phase:
        fstar_max = max(float(c["seq"][1].max()) for c in cache.values()
                        if c["phase"] == phase and len(c["seq"][1]))
        lam_ceiling = LAM_SAFETY / fstar_max
        grid = tuple(x for x in LAM_MAX_GRID if x <= lam_ceiling) or (lam_ceiling,)
        dropped = [x for x in LAM_MAX_GRID if x > lam_ceiling]
        lam_max_grid_by_phase[phase] = grid
        print(f"[{phase}] max f* = {fstar_max:.3f}  ->  ανώτατο επιτρεπτό λ = "
              f"{lam_ceiling:.3f}")
        if dropped:
            print(f"  αφαιρούνται από το grid ως μη φερέγγυα: {dropped}")
        print(f"  grid λmax: {grid}")

    # ---- sanity: constant lambda must BE the shipped fractional Kelly --------
    key = ("Basic", targets[0], "calibrated")
    c = cache[key]
    g_mine, p_mine, _ = simulate_fixed(c["seq"], 0.25, c["T"])
    g_ship, p_ship = simulate_kelly_betting(c["p_hat"], c["odds"], c["y"], c["mask"],
                                            kelly_mult=0.25)
    assert np.allclose(g_mine, g_ship), "fixed lambda must reproduce simulate_kelly_betting"
    assert np.array_equal(p_mine, p_ship), "bet set must match simulate_kelly_betting"
    print("\nsanity ok: σταθερό λ=0,25 == value_betting.simulate_kelly_betting\n")

    # ---- OPTIMISATION: select on a chronological prefix of the BETS ---------
    # Pooled over targets, in the project's primary cell (Basic, calibrated).
    # Nothing outside this file's own layer is re-tuned.
    # Selected SEPARATELY per phase, on that phase's own bets. Pooling the two
    # would make the opening result depend on closing bets it is supposed to be
    # independent of, and would move the already-published closing numbers.
    def split_of(k):
        n = len(cache[k]["seq"][0])
        return int(round(n * TRAIN_FRACTION))

    def pooled(sel_keys, algo, rate, lam_0, lam_max, part):
        """Mean log-growth per bet over the train prefix or the held-out tail,
        pooled across targets. Each target's learner runs on its OWN bets."""
        vals = []
        for k in sel_keys:
            c = cache[k]
            cut = split_of(k)
            growth, _, _ = simulate_learned(c["seq"], c["T"], algo=algo, rate=rate,
                                            lam_0=lam_0, lam_max=lam_max)
            lg = log_growth(c["seq"], growth)
            vals.append(lg[:cut] if part == "train" else lg[cut:])
        return float(np.concatenate(vals).mean())

    rows = []
    chosen = {}
    for phase, phase_targets in targets_by_phase.items():
        sel_keys = [("Basic", b, "calibrated") for b in phase_targets]
        lam_max_grid = lam_max_grid_by_phase[phase]
        chosen[phase] = {}
        for algo in ALGOS:
            print(f"{'=' * 68}\nΒελτιστοποίηση [{phase}]: {algo.upper()}\n{'=' * 68}")
            best = (-np.inf, None, None, None)
            for lam_max in lam_max_grid:
                for lam_0 in LAM_0_GRID:
                    if lam_0 > lam_max:
                        continue
                    rates = list(RATE_GRID)

                    def ev_one(r, _a=algo, _l0=lam_0, _lm=lam_max, _sk=sel_keys,
                               _ph=phase):
                        tr = pooled(_sk, _a, r, _l0, _lm, "train")
                        rows.append({"stage": "sweep", "phase": _ph, "algo": _a,
                                     "rate": r, "lam_0": _l0, "lam_max": _lm,
                                     "train_log_growth": tr,
                                     "held_out_log_growth": pooled(_sk, _a, r, _l0,
                                                                   _lm, "val")})
                        return tr

                    scores = [ev_one(r) for r in rates]
                    if _extend_if_endpoint(rates, ev_one, scores):
                        print(f"      ! ΠΡΟΣΟΧΗ: rate στην ακμή "
                              f"(λ0={lam_0}, λmax={lam_max})")
                    i = int(np.argmax(scores))
                    print(f"  λ0={lam_0:<5g} λmax={lam_max:<4g} -> rate {rates[i]:<8g} "
                          f"train {scores[i]:+.6f}")
                    if scores[i] > best[0]:
                        best = (scores[i], rates[i], lam_0, lam_max)
            chosen[phase][algo] = {"rate": best[1], "lam_0": best[2], "lam_max": best[3]}
            ho = pooled(sel_keys, algo, best[1], best[2], best[3], "val")
            print(f"  ΕΠΙΛΟΓΗ [{phase}] {algo}: rate={best[1]:g}, λ0={best[2]:g}, "
                  f"λmax={best[3]:g}  (train {best[0]:+.6f}, ουρά {ho:+.6f})\n")

    # ---- full comparison on every cell, identical bets everywhere ----------
    methods = {
        "Kelly 1/4 (σταθερό)": lambda c: simulate_fixed(c["seq"], 0.25, c["T"]),
        "Kelly πλήρες (σταθερό)": lambda c: simulate_fixed(c["seq"], 1.0, c["T"]),
    }
    for algo in ALGOS:
        # resolved from the cache entry's own phase, so each target is run with
        # the hyperparameters selected on its own side of the market
        methods[f"OCO λ ({algo.upper()})"] = (
            lambda c, _a=algo: simulate_learned(
                c["seq"], c["T"], algo=_a, **chosen[c["phase"]][_a]))
    def _old_stake_learner(c):
        """The earlier stake-learner, for reference. Its trace is a STAKE, not a
        multiplier, so it reports no mean lambda -- the two are not comparable
        quantities and averaging them into one column would invite exactly that
        confusion."""
        growth, placed, _f_trace = simulate_online_kelly(
            c["p_hat"], c["odds"], c["y"], c["mask"])
        return growth, placed, None

    methods["OCO f (παλαιό)"] = _old_stake_learner

    table, sig_rows, curves, lam_curves = [], [], {}, {}
    for (norm, book, stage), c in cache.items():
        seq = c["seq"]
        if len(seq[0]) < 30:
            continue
        runs = {label: fn(c) for label, fn in methods.items()}
        base_lg = log_growth(seq, runs["Kelly 1/4 (σταθερό)"][0])

        for label, (growth, placed, lam_tr) in runs.items():
            lg = log_growth(seq, growth)
            bank = STARTING_BANKROLL * np.cumprod(growth[c["mask"]])
            res = moving_block_bootstrap_test(lg)
            table.append({
                "normalization": norm, "bookmaker": book, "phase": c["phase"],
                "stage": stage, "method": label, "n_bets": len(seq[0]),
                "final_bankroll": bank[-1] if len(bank) else np.nan,
                "mean_log_growth_per_bet": res["mean_diff"],
                "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                "p_value": res["p_value"], "significant_95": res["significant_95"],
                "mean_lambda": (float(np.nanmean(lam_tr[seq[0]]))
                                if lam_tr is not None else np.nan),
            })
            if norm == "Basic" and stage == "calibrated":
                curves.setdefault(book, {})[label] = bank
                if lam_tr is not None:
                    lam_curves.setdefault(book, {})[label] = lam_tr[seq[0]]

            if label == "Kelly 1/4 (σταθερό)":
                continue
            d = lg - base_lg
            r = moving_block_bootstrap_test(d)
            sig_rows.append({
                "normalization": norm, "bookmaker": book, "phase": c["phase"],
                "stage": stage, "method": label, "vs": "Kelly 1/4",
                "n_bets": len(seq[0]),
                "verdict": ("ισοπαλία" if not r["significant_95"]
                            else (f"{label} better" if r["mean_diff"] > 0
                                  else "Kelly 1/4 better")), **r})

    tbl = pd.DataFrame(table)
    sig = pd.DataFrame(sig_rows)
    pd.set_option("display.width", 240)

    print(f"{'=' * 68}\nΣΥΓΚΡΙΣΗ (Basic, calibrated) -- log-growth ανά στοίχημα\n{'=' * 68}")
    view = tbl[(tbl.normalization == "Basic") & (tbl.stage == "calibrated")]
    print(view[["phase", "bookmaker", "method", "n_bets", "final_bankroll",
                "mean_log_growth_per_bet", "mean_lambda"]]
          .sort_values(["phase", "bookmaker"])
          .to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print(f"\n{'=' * 68}\nΖΕΥΓΑΡΩΜΕΝΟΙ ΕΛΕΓΧΟΙ έναντι Kelly 1/4\n"
          f"ΠΡΟΣΟΧΗ ΣΤΟ ΠΡΟΣΗΜΟ: log-GROWTH, θετικό = καλύτερο από 1/4\n{'=' * 68}")
    print(sig[["normalization", "phase", "bookmaker", "stage", "method", "n_bets",
               "mean_diff", "ci_low", "ci_high", "p_value", "verdict"]]
          .to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print("\n--- σύνοψη ετυμηγοριών, συνολικά ---")
    print(sig.groupby(["method", "verdict"]).size().to_string())
    print("\n--- σύνοψη ετυμηγοριών ανά φάση ---")
    print(sig.groupby(["phase", "method", "verdict"]).size().to_string())

    sweep = pd.DataFrame(rows)
    sp = os.path.join(OUT_DIR, "oco_kelly.csv")
    sgp = os.path.join(OUT_DIR, "oco_kelly_significance.csv")
    pd.concat([sweep, tbl], ignore_index=True).to_csv(sp, index=False)
    sig.to_csv(sgp, index=False)
    print(f"\nSaved: {sp}\nSaved: {sgp}")
    plot_kelly(curves, lam_curves, sweep, os.path.join(OUT_DIR, "oco_kelly.png"))


# ------------------------------------------------------------------ plot --

METHOD_COLORS = {
    "Kelly 1/4 (σταθερό)": COLORS["benchmark"],
    "Kelly πλήρες (σταθερό)": COLORS["muted"],
    "OCO λ (ONS)": COLORS["Hedge"],
    "OCO λ (OGD)": COLORS["FTRL"],
    "OCO f (παλαιό)": COLORS["OGD"],
}
INK, INK_SOFT, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _style(ax):
    ax.grid(True, color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")


def plot_kelly(curves, lam_curves, sweep, out_path):
    books = sorted(curves)
    n = len(books)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 8.4), facecolor=SURFACE,
                             squeeze=False)
    for j, book in enumerate(books):
        ax = axes[0][j]
        ax.set_facecolor(SURFACE)
        ax.axhline(1.0, color="#c3c2b7", linewidth=1.1, linestyle=":", zorder=1)
        for label, bank in curves[book].items():
            ax.plot(np.arange(1, len(bank) + 1), bank, linewidth=1.8, zorder=2,
                    color=METHOD_COLORS.get(label, INK_SOFT), label=label)
        ax.set_yscale("log")
        ax.set_title(book, color=INK, fontsize=11, pad=8)
        # the bankroll advances over every round the book quoted (it is flat on
        # a no-bet round), NOT over bets -- the two axes below differ in length
        # for exactly that reason
        ax.set_xlabel("γύρος με προσφερόμενη απόδοση", color=INK_SOFT)
        _style(ax)

        ax2 = axes[1][j]
        ax2.set_facecolor(SURFACE)
        ax2.axhline(0.25, color="#c3c2b7", linewidth=1.1, linestyle=":", zorder=1)
        for label, tr in lam_curves.get(book, {}).items():
            if label.startswith("OCO λ"):
                ax2.plot(np.arange(1, len(tr) + 1), tr, linewidth=1.8, zorder=2,
                         color=METHOD_COLORS.get(label, INK_SOFT), label=label)
        ax2.set_ylim(0, None)
        ax2.set_xlabel("αριθμός στοιχήματος", color=INK_SOFT)
        _style(ax2)
    axes[0][0].set_ylabel("κεφάλαιο (log, αρχή 1,0)", color=INK_SOFT)
    axes[1][0].set_ylabel("μαθαινόμενο λ (στικτή: το ισχύον 1/4)", color=INK_SOFT)
    axes[1][0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    handles, labels = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, fontsize=9, ncol=len(labels),
                     loc="lower center", bbox_to_anchor=(0.5, -0.03))
    for t in leg.get_texts():
        t.set_color(INK)
    fig.suptitle("OCO στον πολλαπλασιαστή Kelly: κεφάλαιο και τροχιά του λ "
                 "(Basic, calibrated)", color=INK, fontsize=12.5, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
