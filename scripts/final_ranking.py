"""
The single, fair final ranking.

Tier A only: entries with coverage >= 70% (large enough N — 13.5k-19.2k
rounds — that log-loss differences aren't sampling noise; see results_table.csv
for the excluded low-coverage "Tier B" bookmakers, whose apparent lead is not
trustworthy at 2.5k-7k rounds).

For every Tier-A entry (4 mixture algorithms + 8 bookmakers) we report BOTH:
  - raw log-loss/Brier/RPS/accuracy over the full causal history
  - temperature-scaled CALIBRATED versions, computed the SAME way: one
    causal pass over the full history, with the calibration learning rate
    chosen on a train-only split (see calibration_correction.py) — so the
    calibrated number is not fit to the data it's evaluated on.

This makes raw and calibrated directly comparable on equal footing (same
rounds, same evaluation window) — the earlier calibration_correction.py
only reported calibrated numbers on the val slice, which isn't a fair
head-to-head against the full-history raw numbers in results_table.csv.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sleeping_experts import (
    OUT_DIR,
    load_full_universe, per_round_expert_loss,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping,
    evaluate,
)
from calibration_correction import temperature_calibrate, select_lr_on_train, TRAIN_FRACTION, LR_GRID

COVERAGE_THRESHOLD = 60.0  # Tier A cutoff, % of matches covered

RANK_COLORS = {"algorithm": "#2a78d6", "bookmaker": "#eb6834"}


def calibrate_full_history(probs, y, mask):
    """Select the calibration learning rate on a train-only prefix, then run
    ONE causal pass over the full (train+val) history with that fixed rate —
    same protocol as calibration_correction.py, but we keep the full-history
    result instead of only the val slice."""
    p, yy = probs[mask], y[mask]
    n = len(yy)
    n_train = int(round(n * TRAIN_FRACTION))
    best_c, _ = select_lr_on_train(p[:n_train], yy[:n_train], LR_GRID)
    calibrated, _ = temperature_calibrate(p, yy, lr_scale=best_c)
    return p, calibrated, yy, best_c


def main():
    """Builds the ONE table this project treats as the authoritative ranking:
    for every Tier-A entry (4 algorithms, always; bookmakers only if
    coverage_pct >= COVERAGE_THRESHOLD), computes both raw and
    calibrate_full_history()'s calibrated log-loss/Brier/RPS/accuracy over
    the FULL causal history, sorts by calibrated log-loss, saves the table
    and the raw-vs-calibrated slope-chart plot."""
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    loss = per_round_expert_loss(P, y)

    W_hedge = run_hedge_sleeping(loss, awake, N)
    W_ogd = run_ogd_sleeping(loss, awake, N)
    W_ftrl = run_ftrl_sleeping(loss, awake, N)
    W_uniform = awake / awake.sum(axis=1, keepdims=True)
    algo_weights = {"Hedge": W_hedge, "OGD": W_ogd, "FTRL": W_ftrl, "Uniform average": W_uniform}

    entries = {}
    for name, W in algo_weights.items():
        phat = np.einsum("tn,tnk->tk", W, P)
        entries[name] = (phat, np.ones(T, dtype=bool), "algorithm")

    coverage_pct = awake.mean(axis=0) * 100
    for k, book in enumerate(panel):
        if coverage_pct[k] >= COVERAGE_THRESHOLD:
            entries[book] = (P[:, k, :], awake[:, k], "bookmaker")

    rows = []
    for name, (probs, mask, kind) in entries.items():
        p, calibrated, yy, best_c = calibrate_full_history(probs, y, mask)
        raw_m = evaluate(p, yy)
        cal_m = evaluate(calibrated, yy)
        rows.append({
            "name": name, "type": kind, "coverage_pct": mask.mean() * 100, "rounds": int(mask.sum()),
            "best_lr_scale": best_c,
            "raw_log_loss": raw_m["log_loss"], "calibrated_log_loss": cal_m["log_loss"],
            "raw_brier": raw_m["brier"], "calibrated_brier": cal_m["brier"],
            "raw_rps": raw_m["rps"], "calibrated_rps": cal_m["rps"],
            "raw_accuracy": raw_m["accuracy"], "calibrated_accuracy": cal_m["accuracy"],
        })

    table = pd.DataFrame(rows).sort_values("calibrated_log_loss").reset_index(drop=True)
    table_path = os.path.join(OUT_DIR, "final_ranking_table.csv")
    table.to_csv(table_path, index=False)

    pd.set_option("display.width", 160)
    print(f"\n=== FINAL RANKING — Tier A (coverage >= {COVERAGE_THRESHOLD:.0f}%), sorted by calibrated log-loss ===")
    print(f"    saved to {table_path}\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    plot_slope(table, os.path.join(OUT_DIR, "final_ranking.png"))


def plot_slope(table, out_path):
    """Saves results/final_ranking.png: one horizontal row per Tier-A entry,
    an open circle at its raw log-loss and a filled circle at its calibrated
    log-loss, connected by a short line -- a "slope chart" showing both the
    ranking and how much calibration moved each entry, color-coded by
    algorithm (blue) vs. bookmaker (orange)."""
    df = table.sort_values("calibrated_log_loss", ascending=True).reset_index(drop=True)
    n = len(df)
    fig, ax = plt.subplots(figsize=(9, 0.5 * n + 2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ypos = np.arange(n)[::-1]
    for yp, (_, row) in zip(ypos, df.iterrows()):
        color = RANK_COLORS[row["type"]]
        ax.plot([row["raw_log_loss"], row["calibrated_log_loss"]], [yp, yp],
                 color=color, linewidth=1.6, zorder=2, alpha=0.9)
        ax.scatter([row["raw_log_loss"]], [yp], s=70, facecolor="#fcfcfb",
                    edgecolor=color, linewidth=1.8, zorder=3)
        ax.scatter([row["calibrated_log_loss"]], [yp], s=70, facecolor=color,
                    edgecolor=color, linewidth=1.0, zorder=3)

    ax.set_yticks(ypos)
    ax.set_yticklabels(df["name"], fontsize=9, color="#0b0b0b")
    ax.set_xlabel("Log-loss (full history)", color="#52514e")
    ax.set_title("Final ranking: raw vs. temperature-calibrated log-loss (Tier A, coverage ≥ 70%)",
                  color="#0b0b0b", fontsize=12, pad=12)
    ax.grid(True, axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", left=False)

    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="#52514e", markersize=8, markeredgewidth=1.8, label="raw"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#52514e",
               markeredgecolor="#52514e", markersize=8, label="calibrated"),
        Line2D([0], [0], color=RANK_COLORS["algorithm"], linewidth=2.5, label="algorithm"),
        Line2D([0], [0], color=RANK_COLORS["bookmaker"], linewidth=2.5, label="bookmaker"),
    ]
    legend = ax.legend(handles=legend_elements, frameon=False, loc="upper right", fontsize=8.5)
    for text in legend.get_texts():
        text.set_color("#0b0b0b")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
