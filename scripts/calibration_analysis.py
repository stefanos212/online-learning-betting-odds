"""
Calibration (reliability) analysis.

Question: among all the times an algorithm/bookmaker said "this has ~0.3
probability", did the outcome actually happen ~30% of the time?

We pool all three outcome probabilities (p_H, p_D, p_A) together with their
0/1 realized indicator across every match (so a match contributes 3 points:
one for whether H happened, one for D, one for A), bucket the predicted
probability into fixed-width bins (rounding is unavoidable with continuous
probabilities), and compare the bin's mean predicted probability against the
bin's actual hit frequency. Perfect calibration -> points on the y=x diagonal.

Reuses the sleeping-experts data loading and algorithms from sleeping_experts.py.
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

N_BINS = 10
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)

# near-full-coverage bookmakers only, so every bin has a dense-enough sample
# to be a meaningful comparison rather than sampling noise
REFERENCE_BOOKMAKERS = ["B365C", "BW"]

SERIES_COLORS = {
    **COLORS,
    "B365C": COLORS["benchmark"],  # best full-coverage bookmaker (see results_table.csv)
    "BW": COLORS["muted"],         # a visibly weaker full-coverage bookmaker, for contrast
}


def pooled_prob_outcome_pairs(probs, y, mask=None):
    """Flatten (T,3) probs + (T,) outcome indices into (predicted_prob, hit) pairs,
    one pair per outcome per round -> 3 pairs per match."""
    T = probs.shape[0]
    if mask is None:
        mask = np.ones(T, dtype=bool)
    onehot = np.zeros_like(probs)
    onehot[np.arange(T), y] = 1.0
    return probs[mask].reshape(-1), onehot[mask].reshape(-1)


def calibration_table(p_flat, hit_flat, name):
    """Buckets the flattened (predicted_prob, hit) pairs into N_BINS
    fixed-width bins and computes, per non-empty bin: how many pairs fell in
    it, the mean predicted probability, the actual hit frequency, and their
    gap (actual - predicted; positive = underconfident, negative =
    overconfident). One row per bin, tagged with `name` so results from
    several series can be concatenated into one long table."""
    bin_idx = np.digitize(p_flat, BIN_EDGES[1:-1], right=False)
    rows = []
    for b in range(N_BINS):
        sel = bin_idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        rows.append({
            "series": name,
            "bin": f"[{BIN_EDGES[b]:.1f}, {BIN_EDGES[b + 1]:.1f})",
            "bin_center": (BIN_EDGES[b] + BIN_EDGES[b + 1]) / 2,
            "n": n,
            "mean_predicted": p_flat[sel].mean(),
            "actual_frequency": hit_flat[sel].mean(),
        })
    df = pd.DataFrame(rows)
    df["gap"] = df["actual_frequency"] - df["mean_predicted"]
    return df


def expected_calibration_error(df):
    """ECE: bin-count-weighted average |actual - predicted|."""
    return float(np.average(df["gap"].abs(), weights=df["n"]))


def _style_axes(ax):
    """Shared cosmetic styling (gridlines, spine colors, tick colors) applied
    to every axes in this project's matplotlib figures, so all plots look
    like one consistent system. Re-imported (not redefined) by
    calibration_correction.py."""
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")


def plot_reliability(tables, out_path):
    """Saves results/calibration_reliability.png: two panels, (A) the raw
    reliability diagram (predicted vs. actual, marker size = bin sample
    size), and (B) the same data as a "gap" plot (actual - predicted vs.
    predicted) which is what actually makes the small, favorite-longshot-bias
    differences between series visible -- panel A alone has all curves
    overlapping almost exactly."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6.5), facecolor="#fcfcfb",
        gridspec_kw={"width_ratios": [1, 1.15]},
    )

    # ---- Panel A: the raw reliability diagram (context: overall shape) ----
    ax1.set_facecolor("#fcfcfb")
    ax1.plot([0, 1], [0, 1], color="#898781", linewidth=1.5, linestyle="--",
              zorder=1, label="Perfect calibration")
    for name, df in tables.items():
        sizes = 25 + 220 * (df["n"] / df["n"].max())
        color = SERIES_COLORS[name]
        ax1.plot(df["mean_predicted"], df["actual_frequency"], color=color, linewidth=1.5, zorder=2)
        ax1.scatter(df["mean_predicted"], df["actual_frequency"], s=sizes, color=color,
                     label=name, zorder=3, edgecolor="#fcfcfb", linewidth=1.0)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("Mean predicted probability in bin", color="#52514e")
    ax1.set_ylabel("Actual frequency of the outcome", color="#52514e")
    ax1.set_title("Reliability diagram\n(marker size ∝ predictions in that bin)",
                   color="#0b0b0b", fontsize=12, pad=10)
    _style_axes(ax1)
    legend = ax1.legend(frameon=False, loc="upper left", fontsize=8.5)
    for text in legend.get_texts():
        text.set_color("#0b0b0b")

    # ---- Panel B: calibration gap, zoomed in — this is where the small
    # differences between algorithms/bookmakers actually become visible ----
    ax2.set_facecolor("#fcfcfb")
    ax2.axhline(0, color="#898781", linewidth=1.5, linestyle="--", zorder=1)
    for name, df in tables.items():
        color = SERIES_COLORS[name]
        ax2.plot(df["bin_center"], df["gap"] * 100, color=color, linewidth=1.8,
                  marker="o", markersize=5, zorder=3, label=name)
    ax2.set_xlabel("Mean predicted probability in bin", color="#52514e")
    ax2.set_ylabel("Actual − predicted  (percentage points)", color="#52514e")
    ax2.set_title("Calibration gap, zoomed in\n(negative = overconfident, positive = underconfident)",
                   color="#0b0b0b", fontsize=12, pad=10)
    ax2.set_xlim(-0.02, 1.02)
    _style_axes(ax2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def main():
    """Computes calibration tables + ECE for the 4 algorithms (raw mixtures,
    freshly re-run here) plus the REFERENCE_BOOKMAKERS, saves
    calibration_table.csv (per-bin detail) and the reliability plot. This is
    a DIAGNOSTIC pass -- it's what motivated calibration_correction.py's
    online temperature-scaling fix, and doesn't do any correction itself."""
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    loss = per_round_expert_loss(P, y)

    W_hedge = run_hedge_sleeping(loss, awake, N)
    W_ogd = run_ogd_sleeping(loss, awake, N)
    W_ftrl = run_ftrl_sleeping(loss, awake, N)
    W_uniform = awake / awake.sum(axis=1, keepdims=True)

    algo_weights = {"Hedge": W_hedge, "OGD": W_ogd, "FTRL": W_ftrl, "Uniform average": W_uniform}

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

    full_table = pd.concat(tables.values(), ignore_index=True)
    full_path = os.path.join(OUT_DIR, "calibration_table.csv")
    full_table.to_csv(full_path, index=False)

    ece_table = pd.DataFrame(ece_rows).sort_values("ECE")
    print(f"\n=== Expected Calibration Error (ECE), lower = better — per-bin detail saved to {full_path} ===\n")
    print(ece_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Per-bin detail ===\n")
    pd.set_option("display.width", 140)
    print(full_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    plot_reliability(tables, os.path.join(OUT_DIR, "calibration_reliability.png"))


if __name__ == "__main__":
    main()
