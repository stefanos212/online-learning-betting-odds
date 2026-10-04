"""
IS ANY SERIES SIGNIFICANTLY BETTER AT 1 / X / 2 SPECIFICALLY?

`outcome_class_accuracy.py` already breaks accuracy/log-loss/Brier down by
outcome class, but purely descriptively -- it never asks whether the gaps it
prints are real. This file adds the paired test, exactly as
`accuracy_by_band.py` did for probability bands: stratify, then run the
project's standard moving block bootstrap WITHIN each stratum, on rounds
where both series are defined.

READ THIS BEFORE INTERPRETING ANY PER-CLASS RESULT
--------------------------------------------------
Restricted to the rounds that actually ended in a home win, the log-loss is
-log(p_H). So a series that simply says "home" more often than it should
will WIN that stratum -- and lose the draw and away strata by roughly the
same amount. Per-class log-loss on its own therefore measures TILT as much
as skill, and a series that is genuinely better shows up as better in all
three classes, not one.

That is why this file reports, alongside each per-class comparison, the
same comparison pooled over all rounds, and a `consistent` flag: a per-class
win only counts as skill if the pooled comparison agrees in direction. A
win in one class paired with losses in the others is a systematic lean
toward that outcome, which is a calibration defect, not an advantage.

Three views:
  A. DESCRIPTIVE per class, per series: log-loss, Brier, recall (of all the
     actual home wins, how many did it call as its top pick) and precision
     (of all the times it called home, how often was it right). Recall is
     what outcome_class_accuracy.py reports; precision is not in the project
     anywhere else and is the more natural reading of "better at the ace".
  B. ALGORITHM vs ALGORITHM within each class -- does one of the three OCO
     algorithms own a particular outcome?
  C. OGD vs every Tier-A bookmaker within each class.

SIGN: diff = loss(first) - loss(second), so NEGATIVE means the first-named
series is better (the log-loss convention -- see CLAUDE.md).

Produces:
  - results/outcome_class_comparison_descriptive.csv   (view A)
  - results/outcome_class_comparison_significance.csv  (views B and C)
  - results/outcome_class_comparison.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    OUT_DIR, EPS, COLORS,
    load_full_universe, per_round_expert_loss,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping,
)
from calibration_analysis import _style_axes
from final_ranking import COVERAGE_THRESHOLD, RANK_COLORS
from significance_test import moving_block_bootstrap_test, paired_diff, full_length_raw_logloss

CLASS_NAMES = {0: "1 (Home)", 1: "X (Draw)", 2: "2 (Away)"}
ALGOS = ["Hedge", "OGD", "FTRL", "Uniform average"]


def describe(name, probs, y, mask, kind):
    """Per class: how well this series does on the rounds that ended in that
    class (log-loss, Brier, recall) plus precision, which is measured over
    the rounds where it PICKED that class instead."""
    pick = probs.argmax(axis=1)
    rows = []
    for cls, label in CLASS_NAMES.items():
        actual = mask & (y == cls)          # rounds that ended this way
        picked = mask & (pick == cls)       # rounds where this series called it
        n_actual = int(actual.sum())
        if n_actual == 0:
            continue
        p = probs[actual]
        onehot = np.zeros_like(p)
        onehot[:, cls] = 1.0
        rows.append({
            "series": name, "type": kind, "outcome": label,
            "n_actual": n_actual,
            "log_loss": float(-np.log(np.clip(p[:, cls], EPS, 1.0)).mean()),
            "brier": float(((p - onehot) ** 2).sum(axis=1).mean()),
            "recall": float((pick[actual] == cls).mean()),
            "n_picked": int(picked.sum()),
            "precision": float((y[picked] == cls).mean()) if picked.sum() else np.nan,
            "mean_prob_assigned": float(probs[mask][:, cls].mean()),
        })
    return rows


def compare(label_a, ll_a, label_b, ll_b, y, cls, class_label):
    """Paired bootstrap of (a - b) per-round log-loss, restricted to the
    rounds of one outcome class (cls=None means all rounds pooled)."""
    if cls is None:
        a, b = ll_a, ll_b
    else:
        in_class = (y == cls)
        a = np.where(in_class, ll_a, np.nan)
        b = np.where(in_class, ll_b, np.nan)
    d, n = paired_diff(a, b)
    if n < 200:
        return None
    res = moving_block_bootstrap_test(d)
    verdict = ("tie (not significant)" if not res["significant_95"]
               else (f"{label_a} better" if res["mean_diff"] < 0 else f"{label_b} better"))
    return {"outcome": class_label, "comparison": f"{label_a} vs {label_b}",
            "series_a": label_a, "series_b": label_b, "n": n,
            "verdict": verdict, **res}


def plot_classes(desc, out_path):
    """One panel per outcome class: per-class log-loss, algorithms coloured,
    bookmakers grey -- a dot plot, since these values have no meaningful zero
    (same convention as the rest of the project)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor="#fcfcfb")
    for ax, (cls, label) in zip(axes, CLASS_NAMES.items()):
        sub = desc[desc["outcome"] == label].sort_values("log_loss")
        ypos = np.arange(len(sub))[::-1]
        colors = [COLORS[s] if s in COLORS else RANK_COLORS["bookmaker"] for s in sub["series"]]
        ax.set_facecolor("#fcfcfb")
        ax.scatter(sub["log_loss"], ypos, s=80, color=colors, zorder=3,
                    edgecolor="#fcfcfb", linewidth=1.2)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sub["series"], fontsize=8.5, color="#0b0b0b")
        ax.set_xlabel("Log-loss on rounds of this class", color="#52514e", fontsize=9)
        ax.set_title(label, color="#0b0b0b", fontsize=12, pad=10)
        _style_axes(ax)
        ax.grid(False, axis="y")
    fig.suptitle("Per-outcome-class log-loss (algorithms coloured, bookmakers orange)",
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

    print(f"Class base rates: "
          + ", ".join(f"{lab} {100 * (y == c).mean():.1f}%" for c, lab in CLASS_NAMES.items()))
    print(f"Tier-A entities: {list(entries)}\n")

    # ---- A: descriptive ----
    desc_rows, ll = [], {}
    for name, (probs, mask, kind) in entries.items():
        desc_rows += describe(name, probs, y, mask, kind)
        ll[name] = full_length_raw_logloss(probs, y, T, mask=(None if kind == "algorithm" else mask))
    desc = pd.DataFrame(desc_rows)
    desc_path = os.path.join(OUT_DIR, "outcome_class_comparison_descriptive.csv")
    desc.to_csv(desc_path, index=False)

    pd.set_option("display.width", 200)
    print(f"=== A. Per-class detail — saved to {desc_path} ===\n")
    for cls, label in CLASS_NAMES.items():
        sub = desc[desc["outcome"] == label].sort_values("log_loss")
        print(f"-- {label}  (n={int(sub['n_actual'].max())} rounds ended this way) --")
        print(sub[["series", "type", "log_loss", "brier", "recall", "n_picked", "precision",
                    "mean_prob_assigned"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print()

    # ---- B + C: paired tests, per class and pooled ----
    pairs = [("OGD", "Hedge"), ("OGD", "FTRL"), ("OGD", "Uniform average"),
             ("Hedge", "FTRL")]
    pairs += [("OGD", b) for b, (_, _, kind) in entries.items() if kind == "bookmaker"]

    sig_rows = []
    for a, b in pairs:
        pooled = compare(a, ll[a], b, ll[b], y, None, "ALL")
        if pooled is None:
            continue
        sig_rows.append(pooled)
        for cls, label in CLASS_NAMES.items():
            r = compare(a, ll[a], b, ll[b], y, cls, label)
            if r is None:
                continue
            # a per-class win is only evidence of SKILL if the pooled comparison
            # points the same way; otherwise it is a lean toward that outcome
            r["consistent_with_pooled"] = bool(
                np.sign(r["mean_diff"]) == np.sign(pooled["mean_diff"]))
            sig_rows.append(r)

    sig = pd.DataFrame(sig_rows)
    sig_path = os.path.join(OUT_DIR, "outcome_class_comparison_significance.csv")
    sig.to_csv(sig_path, index=False)

    print(f"\n=== B. Algorithm vs algorithm, per class — saved to {sig_path} ===")
    print("    (negative mean_diff = the FIRST-named series is better)\n")
    algo_pairs = {f"{a} vs {b}" for a, b in pairs[:4]}
    for comp in sorted(algo_pairs):
        sub = sig[sig["comparison"] == comp]
        print(f"-- {comp} --")
        print(sub[["outcome", "n", "mean_diff", "ci_low", "ci_high", "p_value", "verdict"]]
              .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        print()

    print(f"\n=== C. OGD vs each Tier-A bookmaker, per class ===\n")
    for cls_label in ["ALL"] + list(CLASS_NAMES.values()):
        sub = sig[(sig["outcome"] == cls_label) & (sig["series_b"].isin(
            [b for b, (_, _, k) in entries.items() if k == "bookmaker"]))]
        if sub.empty:
            continue
        print(f"-- {cls_label} --")
        print(sub[["series_b", "n", "mean_diff", "p_value", "verdict"]]
              .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        print()

    # ---- the tilt check ----
    per_class = sig[sig["outcome"] != "ALL"]
    wins = per_class[per_class["verdict"].str.endswith("better")]
    inconsistent = wins[~wins["consistent_with_pooled"].astype(bool)]
    print(f"\n=== Tilt check ===")
    print(f"significant per-class results: {len(wins)} of {len(per_class)}")
    print(f"of those, pointing the OPPOSITE way to the pooled comparison "
          f"(i.e. a lean toward that outcome, not skill): {len(inconsistent)}")
    if not inconsistent.empty:
        print(inconsistent[["comparison", "outcome", "mean_diff", "p_value", "verdict"]]
              .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    plot_classes(desc, os.path.join(OUT_DIR, "outcome_class_comparison.png"))


if __name__ == "__main__":
    main()
