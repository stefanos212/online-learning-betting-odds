"""
DOES PER-CLASS ACCURACY CARRY ANY SIGNAL AT ALL?

The pooled accuracy tests in market_devig_comparisons.py came back empty --
0 significant results out of 18, p from 0.098 to 0.985 -- so accuracy could
not separate any two methods over all 76,584 matches. The obvious hope is
that it fails only because it averages three very different problems
together, and that split into 1 / X / 2 it starts to say something. This
file checks that hope, and it is the last piece missing from the outcome-
class work: `outcome_class_accuracy.py` and `outcome_class_comparison.py`
both report per-class accuracy numbers, neither tests them.

Two different quantities, which need two different treatments:

  RECALL ("of the matches that ended in a draw, how many did it call?") is
  pairable: every series is defined on the same stratum, so the per-round
  0/1 'picked this class' indicator can go straight into the project's
  standard paired moving block bootstrap, exactly like log-loss.

  PRECISION ("of the times it called a draw, how often was it right?") is
  NOT pairable, because two series call draws on DIFFERENT matches -- there
  is no common denominator to pair on. Same obstacle as comparing two
  betting strategies that place different bets (see
  value_betting_ogd_vs_ftrl.py). So precision gets a per-series bootstrap
  confidence interval instead, and the comparison that IS meaningful is
  against the class base rate: when a series does make this call, is it
  better than the unconditional frequency of that outcome? That is the
  sharpest form of "does this prediction mean anything".

SIGN WARNING: unlike every log-loss table in this project, higher is BETTER
here. diff = accuracy(first) - accuracy(second), so POSITIVE means the
first-named series is better. The `verdict` column states the winner
outright so nobody has to track that (see CLAUDE.md).

Produces:
  - results/outcome_class_accuracy_recall_tests.csv
  - results/outcome_class_accuracy_precision_ci.csv
  - results/outcome_class_accuracy_test.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    OUT_DIR, COLORS,
    load_full_universe, per_round_expert_loss,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping,
)
from calibration_analysis import _style_axes
from final_ranking import COVERAGE_THRESHOLD, RANK_COLORS
from significance_test import moving_block_bootstrap_test, paired_diff

CLASS_NAMES = {0: "1 (Home)", 1: "X (Draw)", 2: "2 (Away)"}
N_BOOT = 2000
SEED = 12345
MIN_N = 30      # below this a bootstrap CI is not worth printing


def recall_series(probs, y, mask, cls):
    """Full-length 0/1 'did this series pick `cls`' indicator, NaN outside the
    stratum or where the series was asleep -- shaped for paired_diff."""
    pick = probs.argmax(axis=1)
    sel = mask & (y == cls)
    out = np.full(len(y), np.nan)
    out[sel] = (pick[sel] == cls).astype(float)
    return out


def precision_ci(probs, y, mask, cls, rng):
    """Bootstrap CI for precision, plus whether it clears the class base rate.

    Resamples the series' own picks with replacement (an i.i.d. bootstrap,
    not a block one: these picks are scattered across ten seasons rather than
    consecutive, so there are no contiguous runs to preserve)."""
    pick = probs.argmax(axis=1)
    picked = mask & (pick == cls)
    n = int(picked.sum())
    base_rate = float((y[mask] == cls).mean())
    if n < MIN_N:
        return {"n_picked": n, "precision": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "base_rate": base_rate, "beats_base_rate": False, "thin": True}
    hits = (y[picked] == cls).astype(float)
    boot = np.array([rng.choice(hits, size=n, replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_picked": n, "precision": float(hits.mean()),
            "ci_low": float(lo), "ci_high": float(hi), "base_rate": base_rate,
            "beats_base_rate": bool(lo > base_rate), "thin": False}


def plot_results(prec, out_path):
    """One panel per class: each series' precision with its bootstrap CI,
    against the dashed base rate. Whether a bar clears the dashed line IS the
    answer to 'does this call mean anything'."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor="#fcfcfb")
    for ax, (cls, label) in zip(axes, CLASS_NAMES.items()):
        sub = prec[(prec["outcome"] == label) & (~prec["thin"])].sort_values("precision")
        ax.set_facecolor("#fcfcfb")
        if sub.empty:
            continue
        ypos = np.arange(len(sub))
        base = sub["base_rate"].iloc[0]
        ax.axvline(base * 100, color="#898781", linewidth=1.5, linestyle="--", zorder=1)
        for yp, (_, r) in zip(ypos, sub.iterrows()):
            color = COLORS.get(r["series"], RANK_COLORS["bookmaker"])
            ax.plot([r["ci_low"] * 100, r["ci_high"] * 100], [yp, yp],
                     color=color, linewidth=2.0, alpha=0.85, zorder=2)
            ax.scatter([r["precision"] * 100], [yp], s=55, color=color, zorder=3,
                        edgecolor="#fcfcfb", linewidth=1.0)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{r['series']} (n={int(r['n_picked'])})"
                            for _, r in sub.iterrows()], fontsize=8, color="#0b0b0b")
        ax.set_xlabel("Precision % when this class is called", color="#52514e", fontsize=9)
        ax.set_title(f"{label}\nbase rate {base * 100:.1f}% (dashed)",
                      color="#0b0b0b", fontsize=11.5, pad=10)
        _style_axes(ax)
        ax.grid(False, axis="y")
    fig.suptitle("When a series actually calls this outcome, is it better than chance?",
                  color="#0b0b0b", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")


def main():
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    loss = per_round_expert_loss(P, y)

    W = {"Hedge": run_hedge_sleeping(loss, awake, N),
         "OGD": run_ogd_sleeping(loss, awake, N),
         "FTRL": run_ftrl_sleeping(loss, awake, N),
         "Uniform average": awake / awake.sum(axis=1, keepdims=True)}

    entries = {}
    for name, Wm in W.items():
        entries[name] = (np.einsum("tn,tnk->tk", Wm, P), np.ones(T, dtype=bool), "algorithm")
    coverage_pct = awake.mean(axis=0) * 100
    for k, book in enumerate(panel):
        if coverage_pct[k] >= COVERAGE_THRESHOLD:
            entries[book] = (P[:, k, :], awake[:, k], "bookmaker")
    books = [b for b, (_, _, k) in entries.items() if k == "bookmaker"]
    print(f"Tier-A entities: {list(entries)}\n")

    # ---- 1. recall: paired tests ----
    pairs = [("OGD", "Hedge"), ("OGD", "FTRL"), ("OGD", "Uniform average"),
             ("Hedge", "FTRL")] + [("OGD", b) for b in books]

    rec_rows = []
    for cls, label in CLASS_NAMES.items():
        rec = {name: recall_series(probs, y, mask, cls)
               for name, (probs, mask, _) in entries.items()}
        for a, b in pairs:
            d, n = paired_diff(rec[a], rec[b])
            if n < MIN_N:
                continue
            res = moving_block_bootstrap_test(d)
            verdict = ("tie (not significant)" if not res["significant_95"]
                       else (f"{a} better" if res["mean_diff"] > 0 else f"{b} better"))
            rec_rows.append({"outcome": label, "comparison": f"{a} vs {b}",
                              "series_a": a, "series_b": b, "n": n,
                              "higher_is_better": True, "verdict": verdict, **res})

    rec_table = pd.DataFrame(rec_rows)
    rec_path = os.path.join(OUT_DIR, "outcome_class_accuracy_recall_tests.csv")
    rec_table.to_csv(rec_path, index=False)

    pd.set_option("display.width", 200)
    print(f"=== 1. RECALL, paired tests — saved to {rec_path} ===")
    print("    (diff = a - b on the 0/1 'picked this class' indicator; POSITIVE = a better)\n")
    for label in CLASS_NAMES.values():
        sub = rec_table[rec_table["outcome"] == label]
        n_sig = int(sub["significant_95"].sum())
        print(f"-- {label}:  {n_sig} of {len(sub)} comparisons significant --")
        print(sub[["comparison", "n", "mean_diff", "ci_low", "ci_high", "p_value", "verdict"]]
              .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        print()

    # ---- 2. precision: per-series CI vs the base rate ----
    rng = np.random.default_rng(SEED)
    prec_rows = []
    for cls, label in CLASS_NAMES.items():
        for name, (probs, mask, kind) in entries.items():
            prec_rows.append({"series": name, "type": kind, "outcome": label,
                               **precision_ci(probs, y, mask, cls, rng)})
    prec = pd.DataFrame(prec_rows)
    prec_path = os.path.join(OUT_DIR, "outcome_class_accuracy_precision_ci.csv")
    prec.to_csv(prec_path, index=False)

    print(f"\n=== 2. PRECISION vs the class base rate — saved to {prec_path} ===")
    print("    (when this series calls the outcome, is it right more often than the outcome")
    print("     simply happens? CI entirely above the base rate = yes)\n")
    for label in CLASS_NAMES.values():
        sub = prec[(prec["outcome"] == label) & (~prec["thin"])].sort_values(
            "precision", ascending=False)
        if sub.empty:
            continue
        base = sub["base_rate"].iloc[0]
        n_beat = int(sub["beats_base_rate"].sum())
        print(f"-- {label}  (base rate {base * 100:.1f}%):  "
              f"{n_beat} of {len(sub)} series beat it --")
        print(sub[["series", "type", "n_picked", "precision", "ci_low", "ci_high",
                    "beats_base_rate"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print()

    # ---- verdict ----
    total = len(rec_table)
    sig = int(rec_table["significant_95"].sum())
    print("\n=== Does per-class accuracy carry signal? ===")
    print(f"RECALL:    {sig} of {total} paired comparisons significant.")
    beat = prec[~prec["thin"]]
    print(f"PRECISION: {int(beat['beats_base_rate'].sum())} of {len(beat)} series x class "
          f"cells beat their base rate at 95%.")

    plot_results(prec, os.path.join(OUT_DIR, "outcome_class_accuracy_test.png"))


if __name__ == "__main__":
    main()
