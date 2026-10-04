"""
FIXED-SHARE FIX for the "frozen advantage" limitation of plain sleeping-
experts Hedge, found while inspecting results_table.csv: rarely-active
experts (often single-season bookmakers, see sleeping_experts.py's
single_season warning) sometimes end up with a disproportionately large
final Hedge weight, because nothing erodes their weight while they are
ASLEEP -- they only compete (and can only lose weight) during the handful of
rounds they're actually awake.

FIX (Herbster & Warmuth, 1998, "Tracking the best expert" -- the standard
"Fixed-Share" algorithm, adapted here to the sleeping-experts setting):
every time an expert is awake and its (renormalized) weight is computed, we
leak a small fraction `alpha` of that weight toward the UNIFORM distribution
over the currently-awake set, before doing the usual multiplicative Hedge
update. This bounds how much of a "stale" advantage a long-dormant expert
can bring back the moment it wakes up: instead of re-entering at whatever
inflated weight it left with, part of that weight is continuously pulled
back toward parity with its (possibly quite different) current competitors.

alpha=0 reduces exactly to plain run_hedge_sleeping() from sleeping_experts.py
-- used here as a sanity check that this file's implementation is a strict
generalization, not a different algorithm.

Produces:
  - results/fixed_share_table.csv   (overall log-loss for a small alpha grid)
  - results/fixed_share_weights.csv (before/after comparison, per
                                      previously-flagged single-season
                                      bookmaker, of the weight it held the
                                      LAST TIME IT WAS AWAKE -- see
                                      sleeping_experts.last_awake_weight for
                                      why the last row of W is the wrong
                                      thing to read here)
"""

import os
import numpy as np
import pandas as pd

from sleeping_experts import (
    HEDGE_STEP_MULTIPLIER,
    OUT_DIR, EPS, M,
    load_full_universe, per_round_expert_loss, run_hedge_sleeping, evaluate,
    last_awake_weight,
)

ALPHA_GRID = [0.0, 0.005, 0.01, 0.02, 0.05]  # 0.0 = plain Hedge, included as a sanity-check baseline


def run_hedge_fixed_share(loss, awake, N, alpha):
    """Identical to sleeping_experts.run_hedge_sleeping when alpha=0 (verified
    in main() below), plus one extra step when alpha>0: AFTER the usual
    multiplicative update, blend a fraction `alpha` of the awake experts'
    (post-update) persistent weight toward a uniform share of that SAME total
    mass, before moving to the next round. Mass-preserving (the blend doesn't
    change sum(w[a])), and a true no-op at alpha=0 -- unlike renormalizing the
    prediction copy `v` itself, which would silently change the dynamics even
    at alpha=0 (an earlier, buggy version of this function did exactly that;
    keeping this note since it's the kind of mistake that's easy to repeat)."""
    w = np.full(N, 1.0 / N)
    T = loss.shape[0]
    W = np.zeros((T, N))
    for t in range(T):
        a = awake[t]
        v = w[a] / w[a].sum()             # renormalize over the awake subset, for prediction ONLY
        W[t, a] = v
        eta = HEDGE_STEP_MULTIPLIER * np.sqrt(np.log(N) / (t + 1)) / M
        w_new = w[a] * np.exp(-eta * loss[t, a])   # exactly sleeping_experts.run_hedge_sleeping's update
        if alpha > 0:
            k = a.sum()
            total = w_new.sum()
            w_new = (1 - alpha) * w_new + alpha * (total / k)   # leak toward uniform, same total mass
        w[a] = w_new
    return W


def main():
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    loss = per_round_expert_loss(P, y)

    # sanity check: alpha=0 must reproduce plain sleeping-experts Hedge exactly
    W_check = run_hedge_fixed_share(loss, awake, N, alpha=0.0)
    W_plain = run_hedge_sleeping(loss, awake, N)
    assert np.allclose(W_check, W_plain), "alpha=0 should exactly match run_hedge_sleeping()"
    print("Sanity check passed: run_hedge_fixed_share(alpha=0) == run_hedge_sleeping()\n")

    # which bookmakers were flagged single_season by sleeping_experts.py? Recompute
    # here directly (cheap) rather than re-reading results_table.csv, so this
    # script stays self-contained and doesn't depend on run order.
    single_season_books = []
    for k, book in enumerate(panel):
        seasons_k = sorted(set(seasons[awake[:, k]]))
        if len(seasons_k) <= 1:
            single_season_books.append(book)
    print(f"Single-season bookmakers (the ones we expect Fixed-Share to rein in): {single_season_books}")

    rows = []
    weight_rows = []
    for alpha in ALPHA_GRID:
        W = run_hedge_fixed_share(loss, awake, N, alpha)
        phat = np.einsum("tn,tnk->tk", W, P)
        m = evaluate(phat, y)
        rows.append({"alpha": alpha, **m})
        print(f"  alpha={alpha:.3f}   log_loss={m['log_loss']:.4f}   brier={m['brier']:.4f}")

        # the weight each bookmaker held the LAST time it was awake -- NOT W[-1, k],
        # which is 0 for anyone asleep in the final match of the dataset. That
        # reading used to hide the worst frozen-advantage cases this file exists
        # to measure: the four 2024/25-only bookmakers (1XB, 1XBC, BF, BFC) all
        # reported 0.0 at every alpha, while actually sitting on ~0.16-0.18.
        w_last = last_awake_weight(W, awake)
        for book in single_season_books:
            k = panel.index(book)
            weight_rows.append({"alpha": alpha, "bookmaker": book, "weight_last_awake": w_last[k]})

    table = pd.DataFrame(rows)
    table_path = os.path.join(OUT_DIR, "fixed_share_table.csv")
    table.to_csv(table_path, index=False)
    print(f"\n=== Overall log-loss vs. alpha (0.0 = plain Hedge) — saved to {table_path} ===\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    weight_table = pd.DataFrame(weight_rows).pivot(index="bookmaker", columns="alpha", values="weight_last_awake")
    weight_path = os.path.join(OUT_DIR, "fixed_share_weights.csv")
    weight_table.to_csv(weight_path)
    print(f"\n=== Hedge weight of each single-season bookmaker the LAST time it was awake, "
          f"by alpha — saved to {weight_path} ===\n")
    print(f"(alpha=0.0 column = plain sleeping-experts Hedge from sleeping_experts.py, i.e. the ORIGINAL problem)\n")
    print(weight_table.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
