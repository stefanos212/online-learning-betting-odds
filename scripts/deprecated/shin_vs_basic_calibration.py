"""
Does Shin normalization actually produce better-CALIBRATED raw probabilities
than basic proportional normalization -- i.e. does it do what it's supposed
to do? shin_vs_basic_comparison.py answers "which gets the lower loss";
this file answers the more direct mechanistic question calibration_analysis.py
originally asked of the basic-normalized panel: among all the times a series
said "~30% probability", did the outcome happen ~30% of the time?

Reuses calibration_analysis.py's binning machinery unchanged
(pooled_prob_outcome_pairs, calibration_table, expected_calibration_error,
_style_axes) on RAW (uncalibrated) probabilities from both odds_long.csv
(basic normalization) and odds_long_shin.csv (Shin normalization), for the
same 4 algorithms + REFERENCE_BOOKMAKERS as calibration_analysis.py. Since
Shin's method is specifically designed to correct the favorite-longshot
bias, the expectation is a smaller ECE and a flatter calibration-gap curve
under Shin -- this is what actually gets checked here rather than assumed.

Produces:
  - results/shin_vs_basic_calibration_table.csv  (per-bin detail, both
    normalizations)
  - results/shin_vs_basic_calibration_ece.csv    (ECE summary, both
    normalizations, + the reduction from Basic to Shin)
  - results/shin_vs_basic_calibration_reliability.png  (two side-by-side
    gap plots, Basic vs Shin, same y-axis scale for a direct comparison)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    DATA_DIR, OUT_DIR,
    per_round_expert_loss, run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping,
)
from calibration_analysis import (
    REFERENCE_BOOKMAKERS, SERIES_COLORS,
    pooled_prob_outcome_pairs, calibration_table, expected_calibration_error, _style_axes,
)
from shin_vs_basic_comparison import load_universe_from

SERIES_ORDER = ["Hedge", "OGD", "FTRL", "Uniform average"] + REFERENCE_BOOKMAKERS


def calibration_for_dataset(panel, P, awake, y):
    """Raw (uncalibrated) calibration tables for the 4 algorithms +
    REFERENCE_BOOKMAKERS on one panel -- same entity set as
    calibration_analysis.main(), just parameterized over an arbitrary
    (panel, P, awake, y) instead of always calling load_full_universe()."""
    T, N, _ = P.shape
    loss = per_round_expert_loss(P, y)
    algo_weights = {
        "Hedge": run_hedge_sleeping(loss, awake, N),
        "OGD": run_ogd_sleeping(loss, awake, N),
        "FTRL": run_ftrl_sleeping(loss, awake, N),
        "Uniform average": awake / awake.sum(axis=1, keepdims=True),
    }

    tables, ece_rows = {}, []
    for name, W in algo_weights.items():
        phat = np.einsum("tn,tnk->tk", W, P)
        p_flat, hit_flat = pooled_prob_outcome_pairs(phat, y)
        df = calibration_table(p_flat, hit_flat, name)
        tables[name] = df
        ece_rows.append({"series": name, "type": "algorithm", "n_total": len(p_flat),
                          "ECE": expected_calibration_error(df)})

    for book in REFERENCE_BOOKMAKERS:
        k = panel.index(book)
        p_flat, hit_flat = pooled_prob_outcome_pairs(P[:, k, :], y, mask=awake[:, k])
        df = calibration_table(p_flat, hit_flat, book)
        tables[book] = df
        ece_rows.append({"series": book, "type": "bookmaker", "n_total": len(p_flat),
                          "ECE": expected_calibration_error(df)})

    return tables, ece_rows


def plot_reliability_comparison(tables_by_label, out_path):
    """Saves shin_vs_basic_calibration_reliability.png: two side-by-side
    calibration-GAP panels (the zoomed panel from
    calibration_analysis.plot_reliability -- what actually makes small
    favorite-longshot-bias differences visible), one for Basic normalization
    and one for Shin, sharing a y-axis so the flattening (or not) is
    directly comparable at a glance. Same series colors as
    calibration_analysis.py (SERIES_COLORS) -- no new colors invented."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="#fcfcfb", sharey=True)

    for ax, label in zip(axes, ["Basic", "Shin"]):
        ax.set_facecolor("#fcfcfb")
        ax.axhline(0, color="#898781", linewidth=1.5, linestyle="--", zorder=1)
        for name in SERIES_ORDER:
            df = tables_by_label[label][name]
            color = SERIES_COLORS[name]
            ax.plot(df["bin_center"], df["gap"] * 100, color=color, linewidth=1.8,
                     marker="o", markersize=5, zorder=3, label=name)
        ax.set_xlabel("Mean predicted probability in bin", color="#52514e")
        ax.set_title(f"{label} normalization", color="#0b0b0b", fontsize=12, pad=10)
        ax.set_xlim(-0.02, 1.02)
        _style_axes(ax)

    axes[0].set_ylabel("Actual − predicted  (percentage points)", color="#52514e")
    legend = axes[1].legend(frameon=False, loc="upper left", fontsize=8.5)
    for text in legend.get_texts():
        text.set_color("#0b0b0b")

    fig.suptitle("Calibration gap, zoomed in (negative = overconfident, positive = underconfident)",
                  color="#0b0b0b", fontsize=12.5, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")


def main():
    """Builds raw calibration tables + ECE for the 4 algorithms +
    REFERENCE_BOOKMAKERS on both odds_long.csv and odds_long_shin.csv,
    saves the per-bin table, an ECE-reduction summary, and the side-by-side
    gap-comparison plot."""
    paths = {"Basic": os.path.join(DATA_DIR, "odds_long.csv"),
              "Shin": os.path.join(DATA_DIR, "odds_long_shin.csv")}

    tables_by_label, ece_by_label = {}, {}
    for label, path in paths.items():
        panel, P, awake, y, dates, seasons = load_universe_from(path)
        tables, ece_rows = calibration_for_dataset(panel, P, awake, y)
        tables_by_label[label] = tables
        ece_by_label[label] = pd.DataFrame(ece_rows).set_index("series")
        print(f"{label}: computed calibration tables for {list(tables.keys())}")

    # ---- per-bin detail, both normalizations, one CSV ----
    all_bins = []
    for label, tables in tables_by_label.items():
        for name, df in tables.items():
            tagged = df.copy()
            tagged["normalization"] = label
            all_bins.append(tagged)
    bin_table = pd.concat(all_bins, ignore_index=True)
    bin_path = os.path.join(OUT_DIR, "shin_vs_basic_calibration_table.csv")
    bin_table.to_csv(bin_path, index=False)
    print(f"\nSaved: {bin_path}")

    # ---- ECE summary: Basic vs Shin, + the reduction ----
    ece = ece_by_label["Basic"][["type", "n_total", "ECE"]].rename(columns={"ECE": "ECE_basic"})
    ece["ECE_shin"] = ece_by_label["Shin"]["ECE"]
    ece["ECE_reduction"] = ece["ECE_basic"] - ece["ECE_shin"]  # positive = Shin better calibrated
    ece["ECE_reduction_pct"] = (ece["ECE_reduction"] / ece["ECE_basic"] * 100).round(1)
    ece = ece.sort_values("ECE_basic")
    ece_path = os.path.join(OUT_DIR, "shin_vs_basic_calibration_ece.csv")
    ece.to_csv(ece_path)

    pd.set_option("display.width", 140)
    print(f"\n=== Expected Calibration Error: Basic vs. Shin "
          f"(positive ECE_reduction = Shin is BETTER calibrated) ===\n")
    print(ece.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved: {ece_path}")

    plot_reliability_comparison(tables_by_label, os.path.join(OUT_DIR, "shin_vs_basic_calibration_reliability.png"))


if __name__ == "__main__":
    main()
