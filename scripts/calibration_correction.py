"""
Online temperature scaling: a second, calibration-only OCO stage stacked on
top of each mixture (Hedge / OGD / FTRL / Uniform) — and, for comparison, on
top of two individual bookmakers (B365C, BW).

Why temperature scaling: calibration_analysis.py found a favorite-longshot
bias shared by every series (mixtures included) — low predicted probabilities
are systematically too high, high predicted probabilities are systematically
too low. That is exactly the "underconfident" pattern that scaling
probabilities away from the middle corrects, i.e. exponent s = 1/T > 1:

    calibrated_k = phat_k^s / sum_j phat_j^s

Why this is still OCO: with log_phat = log(phat) (clipped away from 0),
loss(s) = -s*log_phat[y] + log( sum_k exp(s * log_phat[k]) )
is a log-sum-exp of a linear function of s, hence CONVEX in s. So we can fit
s online with plain projected OGD on a 1-D interval — no new machinery needed,
this is the same OCO recipe as the rest of the thesis, just with a 1-D decision
variable instead of a simplex.

Train/val protocol (chronological, no leakage):
  - first TRAIN_FRACTION of each series' own rounds = train
  - the OGD learning-rate scale c is selected by train-period log-loss only
    (a small grid, evaluated on a train-only run of the calibrator)
  - the reported numbers are then a SEPARATE, single causal run over the full
    (train+val) timeline with that fixed c, with metrics computed on the held-
    out val rounds only (the val-period numbers never influenced the choice of c)

Produces:
  - results/calibration_correction_table.csv  (raw vs calibrated x train/val
                                                 x every series)
  - results/calibration_correction.png        (val-only reliability + gap,
                                                 raw dashed vs calibrated solid,
                                                 for the 4 mixture algorithms)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    OUT_DIR, EPS, COLORS,
    load_full_universe, per_round_expert_loss,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping,
    evaluate,
)
from calibration_analysis import (
    REFERENCE_BOOKMAKERS, SERIES_COLORS,
    pooled_prob_outcome_pairs, calibration_table, expected_calibration_error,
    _style_axes,
)

TRAIN_FRACTION = 0.75
LR_GRID = (0.05, 0.15, 0.4, 1.0)   # candidate scales for eta_t = c / sqrt(t+1)
S_MIN, S_MAX = 0.2, 5.0             # bounds on s = 1/T


# ------------------------------------------------------- online calibrator --

def temperature_calibrate(probs, y, lr_scale):
    """Causal online temperature scaling (exponent s = 1/T) via projected OGD
    on the convex loss(s) = -s*log_phat[y] + log(sum_k exp(s*log_phat[k]))."""
    T = len(y)
    log_phat = np.log(np.clip(probs, EPS, 1.0))
    s = 1.0  # start at T=1, i.e. "no correction"
    calibrated = np.zeros((T, 3))
    s_hist = np.zeros(T)
    for t in range(T):
        s_hist[t] = s
        exps = np.exp(s * log_phat[t])
        w = exps / exps.sum()
        calibrated[t] = w
        grad = -log_phat[t, y[t]] + np.dot(w, log_phat[t])  # d loss/ds at current s
        eta = lr_scale / np.sqrt(t + 1)
        s = np.clip(s - eta * grad, S_MIN, S_MAX)
    return calibrated, s_hist


def select_lr_on_train(probs_train, y_train, lr_grid):
    """Tries each candidate learning-rate scale in `lr_grid`, running the
    calibrator causally on the TRAIN prefix only, and returns whichever gives
    the lowest train-period log-loss. This is the "no leakage" step: nothing
    here ever looks at the val split, so choosing a rate this way and then
    running once more over the full timeline (see run_series below) can't
    have benefited from future information."""
    best_c, best_loss = None, np.inf
    for c in lr_grid:
        calibrated, _ = temperature_calibrate(probs_train, y_train, lr_scale=c)
        idx = np.arange(len(y_train))
        loss = -np.log(np.clip(calibrated[idx, y_train], EPS, 1.0)).mean()
        if loss < best_loss:
            best_loss, best_c = loss, c
    return best_c, best_loss


# ------------------------------------------------------------- per-series --

def run_series(name, probs, y, mask=None):
    """probs, y: full (T,3)/(T,) arrays; mask (optional) restricts to the
    rounds this series actually participated in (e.g. a partial-coverage
    bookmaker), preserving chronological order within that subset."""
    T = len(y)
    if mask is None:
        mask = np.ones(T, dtype=bool)
    p, yy = probs[mask], y[mask]
    n = len(yy)
    n_train = int(round(n * TRAIN_FRACTION))
    split = np.arange(n) < n_train  # True = train, False = val

    best_c, train_loss_at_best_c = select_lr_on_train(p[:n_train], yy[:n_train], LR_GRID)
    calibrated, s_hist = temperature_calibrate(p, yy, lr_scale=best_c)

    rows = []
    for split_name, sel in [("train", split), ("val", ~split)]:
        raw_m = evaluate(p, yy, mask=sel)
        cal_m = evaluate(calibrated, yy, mask=sel)
        rows.append({"series": name, "split": split_name, "stage": "raw", **raw_m})
        rows.append({"series": name, "split": split_name, "stage": "calibrated", **cal_m})

    final_T = 1.0 / s_hist[-1]
    val_sel = ~split
    return rows, best_c, final_T, (p[val_sel], calibrated[val_sel], yy[val_sel])


# --------------------------------------------------------------- plotting --

def plot_val_comparison(val_data, out_path):
    """Saves results/calibration_correction.png: same two-panel design as
    calibration_analysis.plot_reliability, but restricted to the held-out val
    split and showing raw (dashed, faded) vs. calibrated (solid) for each of
    the 4 algorithms, so the correction's effect is directly visible."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6.5), facecolor="#fcfcfb",
        gridspec_kw={"width_ratios": [1, 1.15]},
    )

    ax1.set_facecolor("#fcfcfb")
    ax1.plot([0, 1], [0, 1], color="#898781", linewidth=1.5, linestyle="--", zorder=1)
    ax2.set_facecolor("#fcfcfb")
    ax2.axhline(0, color="#898781", linewidth=1.5, linestyle="--", zorder=1)

    for name, (p_raw, p_cal, yy) in val_data.items():
        color = SERIES_COLORS[name]
        for stage, probs, ls, alpha in [("raw", p_raw, "--", 0.55), ("calibrated", p_cal, "-", 1.0)]:
            p_flat, hit_flat = pooled_prob_outcome_pairs(probs, yy)
            df = calibration_table(p_flat, hit_flat, name)
            label = f"{name} ({stage})"
            ax1.plot(df["mean_predicted"], df["actual_frequency"], color=color,
                      linewidth=1.8, linestyle=ls, alpha=alpha, zorder=2, label=label)
            ax2.plot(df["bin_center"], df["gap"] * 100, color=color, linewidth=1.8,
                      linestyle=ls, alpha=alpha, marker="o", markersize=4, zorder=2, label=label)

    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("Mean predicted probability in bin", color="#52514e")
    ax1.set_ylabel("Actual frequency of the outcome", color="#52514e")
    ax1.set_title("Reliability, held-out val split\ndashed = raw, solid = calibrated",
                   color="#0b0b0b", fontsize=12, pad=10)
    _style_axes(ax1)
    legend1 = ax1.legend(frameon=False, loc="upper left", fontsize=7.5, ncol=1)
    for text in legend1.get_texts():
        text.set_color("#0b0b0b")

    ax2.set_xlabel("Mean predicted probability in bin", color="#52514e")
    ax2.set_ylabel("Actual − predicted  (percentage points)", color="#52514e")
    ax2.set_title("Calibration gap, held-out val split\n(negative = overconfident, positive = underconfident)",
                   color="#0b0b0b", fontsize=12, pad=10)
    ax2.set_xlim(-0.02, 1.02)
    _style_axes(ax2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


# ---------------------------------------------------------------- main -----

def main():
    """Runs run_series() (fit + train/val-evaluate the temperature
    calibrator) for each of the 4 algorithms and the 2 REFERENCE_BOOKMAKERS,
    saves the full raw-vs-calibrated x train-vs-val table, prints the VAL-only
    honest comparison (log-loss and ECE), and saves the plot. NOTE: the
    metrics here are computed on the VAL split only -- for a fair full-history
    comparison against results_table.csv's raw numbers, see final_ranking.py,
    which reruns this same calibration causally over the ENTIRE timeline."""
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    loss = per_round_expert_loss(P, y)

    W_hedge = run_hedge_sleeping(loss, awake, N)
    W_ogd = run_ogd_sleeping(loss, awake, N)
    W_ftrl = run_ftrl_sleeping(loss, awake, N)
    W_uniform = awake / awake.sum(axis=1, keepdims=True)
    algo_weights = {"Hedge": W_hedge, "OGD": W_ogd, "FTRL": W_ftrl, "Uniform average": W_uniform}

    all_rows, summary_rows, val_data = [], [], {}

    for name, W in algo_weights.items():
        phat = np.einsum("tn,tnk->tk", W, P)
        rows, best_c, final_T, val_tuple = run_series(name, phat, y)
        all_rows += rows
        summary_rows.append({"series": name, "type": "algorithm", "best_lr_scale": best_c, "final_T": final_T})
        val_data[name] = val_tuple

    for book in REFERENCE_BOOKMAKERS:
        k = panel.index(book)
        rows, best_c, final_T, val_tuple = run_series(book, P[:, k, :], y, mask=awake[:, k])
        all_rows += rows
        summary_rows.append({"series": book, "type": "bookmaker", "best_lr_scale": best_c, "final_T": final_T})
        val_data[book] = val_tuple

    table = pd.DataFrame(all_rows)
    table_path = os.path.join(OUT_DIR, "calibration_correction_table.csv")
    table.to_csv(table_path, index=False)

    summary = pd.DataFrame(summary_rows)
    pd.set_option("display.width", 140)
    print(f"\n=== Chosen temperature per series (selected on TRAIN only) — full table saved to {table_path} ===\n")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Raw vs calibrated, VAL split only (the honest, held-out comparison) ===\n")
    val_table = table[table["split"] == "val"].drop(columns=["split"])
    print(val_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ECE on the val split specifically, since the printed metrics above don't include it
    print("\n=== Expected Calibration Error (ECE), VAL split, raw vs calibrated ===\n")
    ece_rows = []
    for name, (p_raw, p_cal, yy) in val_data.items():
        for stage, probs in [("raw", p_raw), ("calibrated", p_cal)]:
            p_flat, hit_flat = pooled_prob_outcome_pairs(probs, yy)
            df = calibration_table(p_flat, hit_flat, name)
            ece_rows.append({"series": name, "stage": stage, "ECE": expected_calibration_error(df)})
    ece_table = pd.DataFrame(ece_rows).sort_values(["series", "stage"])
    print(ece_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    algo_val_data = {k: v for k, v in val_data.items() if k in algo_weights}
    plot_val_comparison(algo_val_data, os.path.join(OUT_DIR, "calibration_correction.png"))


if __name__ == "__main__":
    main()
