"""
PER-OUTCOME-CLASS (1 / X / 2) BREAKDOWN.

Every metric elsewhere in this project (results_table.csv, final_ranking.py,
etc.) averages over all three outcome classes together. But football
forecasting has a well-known asymmetry: draws are notoriously hard to call
as the single most likely outcome -- a draw is rarely ANY forecaster's
highest probability, even for matches that end up drawn -- so a series with
a great POOLED log-loss can still have near-zero accuracy specifically at
picking draws. This file answers "which algorithm/bookmaker is actually best
at each of the three specific bet types" rather than the pooled average,
across ALL matches (not split by league, unlike contextual_experts.py).

Restricts the bookmaker side to Tier A (coverage >= 70%, same threshold as
final_ranking.py) -- per-class sample sizes are already smaller than the
pooled ones (draws are ~26% of matches), so low-coverage bookmakers would be
even less trustworthy here than in the pooled comparison.

Produces:
  - results/outcome_class_accuracy_table.csv
  - results/outcome_class_accuracy.png   (3 panels: accuracy at picking H/D/A)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    OUT_DIR, EPS,
    load_full_universe, per_round_expert_loss,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping,
)

COVERAGE_THRESHOLD = 60.0  # keep in step with final_ranking.COVERAGE_THRESHOLD
OUTCOME_LABELS = {0: "1 (Home win)", 1: "X (Draw)", 2: "2 (Away win)"}
TYPE_COLORS = {"algorithm": "#2a78d6", "bookmaker": "#eb6834"}


def per_class_metrics(probs, y, mask=None):
    """probs: (T,3) predicted probabilities, y: (T,) true outcome index (0/1/2).
    One row of metrics per outcome class, restricted to rounds whose TRUE
    outcome was that class (and, if given, `mask`) -- i.e. per-class RECALL:
    of all actual home wins, what fraction did this series correctly call as
    its single most likely outcome?"""
    T = len(y)
    if mask is None:
        mask = np.ones(T, dtype=bool)
    rows = []
    for cls in range(3):
        sel = mask & (y == cls)
        n = int(sel.sum())
        if n == 0:
            continue
        p = probs[sel]
        pred = p.argmax(axis=1)
        acc = float((pred == cls).mean())
        ll = float(-np.log(np.clip(p[:, cls], EPS, 1.0)).mean())
        onehot = np.zeros_like(p)
        onehot[:, cls] = 1.0
        brier = float(((p - onehot) ** 2).sum(axis=1).mean())
        rows.append({"outcome": OUTCOME_LABELS[cls], "n": n, "accuracy": acc,
                      "log_loss": ll, "brier": brier})
    return rows


def plot_by_class(table, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.5), facecolor="#fcfcfb")
    for ax, outcome in zip(axes, OUTCOME_LABELS.values()):
        sub = table[table["outcome"] == outcome].sort_values("accuracy")
        colors = [TYPE_COLORS[t] for t in sub["type"]]
        ypos = np.arange(len(sub))

        ax.set_facecolor("#fcfcfb")
        ax.barh(ypos, sub["accuracy"] * 100, color=colors, height=0.65, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sub["name"], fontsize=7.5, color="#0b0b0b")
        ax.set_xlabel("Accuracy % (picked as single most likely outcome)", color="#52514e", fontsize=8.5)
        ax.set_title(outcome, color="#0b0b0b", fontsize=13, pad=10)
        ax.grid(True, axis="x", color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors="#898781")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=TYPE_COLORS["algorithm"], linewidth=8, label="algorithm"),
        Line2D([0], [0], color=TYPE_COLORS["bookmaker"], linewidth=8, label="bookmaker"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle("Accuracy at picking each outcome as the single most likely one",
                  color="#0b0b0b", fontsize=13, y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")


def main():
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    loss = per_round_expert_loss(P, y)

    W_hedge = run_hedge_sleeping(loss, awake, N)
    W_ogd = run_ogd_sleeping(loss, awake, N)
    W_ftrl = run_ftrl_sleeping(loss, awake, N)
    W_uniform = awake / awake.sum(axis=1, keepdims=True)
    algo_weights = {"Hedge": W_hedge, "OGD": W_ogd, "FTRL": W_ftrl, "Uniform average": W_uniform}

    rows = []
    for name, W in algo_weights.items():
        phat = np.einsum("tn,tnk->tk", W, P)
        for r in per_class_metrics(phat, y):
            rows.append({"name": name, "type": "algorithm", **r})

    coverage_pct = awake.mean(axis=0) * 100
    for k, book in enumerate(panel):
        if coverage_pct[k] >= COVERAGE_THRESHOLD:
            for r in per_class_metrics(P[:, k, :], y, mask=awake[:, k]):
                rows.append({"name": book, "type": "bookmaker", **r})

    table = pd.DataFrame(rows)
    table_path = os.path.join(OUT_DIR, "outcome_class_accuracy_table.csv")
    table.to_csv(table_path, index=False)

    pd.set_option("display.width", 140)
    print(f"\n=== Per-outcome-class accuracy/log-loss (Tier A only) — saved to {table_path} ===\n")
    for outcome in OUTCOME_LABELS.values():
        sub = table[table["outcome"] == outcome].sort_values("accuracy", ascending=False)
        print(f"\n--- {outcome} ---")
        print(sub.drop(columns=["outcome"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    plot_by_class(table, os.path.join(OUT_DIR, "outcome_class_accuracy.png"))


if __name__ == "__main__":
    main()
