"""
PER-SEASON (instead of global) temperature calibration.

Motivation (see temporal_trends.py and the theoretical discussion this file
implements): calibration_correction.py / final_ranking.py's
calibrate_full_history() runs ONE continuous causal OGD fit for the
temperature exponent s=1/T across the full 10-year history, with a
STATIC-regret learning-rate schedule eta_t = c/sqrt(t+1). That schedule's
step size shrinks over the whole decade, so by the later seasons the fitted
s is essentially frozen near a long-run compromise value -- it cannot track
a genuinely time-varying bias. temporal_trends.py found the market's
overround changes significantly season to season, so a single fixed
correction plausibly fits none of the individual seasons as well as a
per-season one could -- which is what this file tests. (An earlier version
of this docstring motivated the file by claiming global calibration
SIGNIFICANTLY HURT log-loss on the 10-year dataset. That was wrong: it came
from misreading significance_test.py's sign convention -- see CLAUDE.md.
Global calibration significantly HELPS every series; the open question is
only whether a per-season fit helps even more.)

This file resets the round counter (hence eta_t) at the start of EVERY
season, giving each season its own fast-adapting calibration phase, instead
of inheriting an already-tiny step size from prior years. Concretely: reuses
temperature_calibrate()/select_lr_on_train() from calibration_correction.py
completely unmodified, just called once per season on that season's own
slice (same trick as contextual_experts.py's per-league independent fits,
applied along the time axis instead of the league axis).

Trade-off to watch: each season has ~7,658 rounds (76,584 / 10) versus the
full history's 76,584 -- much less data to select the learning rate and fit
s. But this is a single SCALAR parameter (unlike per-league OGD's full
N-dimensional weight vector in contextual_experts.py, which lost to pooling
partly because of exactly this sample-size problem), so it should need much
less data to fit well. Whether the better within-season fit outweighs the
smaller sample and the 10x-repeated cold-start cost is an empirical
question -- this file measures it directly, both against RAW (uncalibrated)
and against the existing GLOBAL calibration, on equal footing (same
full-history evaluation as final_ranking.py).

Produces:
  - results/calibration_per_season_table.csv   (raw vs global-calibrated vs
                                                  per-season-calibrated, full
                                                  history, for the 4
                                                  algorithms + Tier-A bookmakers)
  - results/calibration_per_season_significance.csv  (bootstrap tests:
                                                  per-season vs raw, and
                                                  per-season vs global)
  - results/calibration_per_season.png         (per-season fitted
                                                  temperature T, one line per
                                                  series -- direct evidence of
                                                  how much T actually moves
                                                  season to season)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    OUT_DIR, EPS, COLORS,
    load_full_universe, per_round_expert_loss,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping, evaluate,
)
from calibration_correction import temperature_calibrate, select_lr_on_train, TRAIN_FRACTION, LR_GRID
from final_ranking import calibrate_full_history, COVERAGE_THRESHOLD
from significance_test import moving_block_bootstrap_test, paired_diff

SERIES_COLORS = {**COLORS, "B365C": COLORS["benchmark"]}


def calibrate_per_season(probs, y, mask, seasons, season_order):
    """Same train/val protocol as calibrate_full_history(), but run
    INDEPENDENTLY once per season: the round counter (and so eta_t) resets
    to 0 at the start of every season, instead of continuing from a decade's
    worth of accumulated rounds. Returns a full-length calibrated array
    (aligned to the original T-length index, NaN where masked out) plus a
    per-season dict of the fitted final temperature T=1/s."""
    T = len(y)
    calibrated_full = np.full((T, 3), np.nan)
    final_T_by_season = {}

    for s in season_order:
        sel = mask & (seasons == s)
        if not sel.any():
            continue
        p, yy = probs[sel], y[sel]
        n = len(yy)
        n_train = max(1, int(round(n * TRAIN_FRACTION)))
        best_c, _ = select_lr_on_train(p[:n_train], yy[:n_train], LR_GRID)
        calibrated, s_hist = temperature_calibrate(p, yy, lr_scale=best_c)
        calibrated_full[sel] = calibrated
        final_T_by_season[s] = 1.0 / s_hist[-1]

    return calibrated_full, final_T_by_season


def plot_temperatures(temps_table, season_order, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axhline(1.0, color="#898781", linewidth=1.2, linestyle="--", zorder=1, label="T=1 (no correction)")

    xpos = np.arange(len(season_order))
    for series in temps_table["series"].unique():
        sub = temps_table[temps_table["series"] == series].set_index("season").reindex(season_order)
        color = SERIES_COLORS.get(series, "#898781")
        ax.plot(xpos, sub["T"], color=color, linewidth=1.8, marker="o", markersize=4, label=series, zorder=3)

    ax.set_xticks(xpos)
    ax.set_xticklabels(season_order, rotation=45, fontsize=8)
    ax.set_ylabel("Fitted temperature T (per season)", color="#52514e")
    ax.set_title("Per-season fitted temperature -- how much does T actually move?",
                  color="#0b0b0b", fontsize=12, pad=12)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")
    legend = ax.legend(frameon=False, loc="best", fontsize=8)
    for text in legend.get_texts():
        text.set_color("#0b0b0b")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def main():
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    season_order = sorted(set(seasons))
    loss = per_round_expert_loss(P, y)

    W_hedge = run_hedge_sleeping(loss, awake, N)
    W_ogd = run_ogd_sleeping(loss, awake, N)
    W_ftrl = run_ftrl_sleeping(loss, awake, N)
    W_uniform = awake / awake.sum(axis=1, keepdims=True)
    algo_weights = {"Hedge": W_hedge, "OGD": W_ogd, "FTRL": W_ftrl, "Uniform average": W_uniform}

    entries = {}
    for name, W in algo_weights.items():
        phat = np.einsum("tn,tnk->tk", W, P)
        entries[name] = (phat, np.ones(T, dtype=bool))
    coverage_pct = awake.mean(axis=0) * 100
    for k, book in enumerate(panel):
        if coverage_pct[k] >= COVERAGE_THRESHOLD:
            entries[book] = (P[:, k, :], awake[:, k])

    rows, temp_rows, sig_rows = [], [], []
    raw_ll, global_ll, season_ll = {}, {}, {}

    for name, (probs, mask) in entries.items():
        print(f"  processing {name}...")
        # raw
        idx = np.arange(T)
        raw_full = -np.log(np.clip(probs[idx, y], EPS, 1.0))
        raw_full = np.where(mask, raw_full, np.nan)

        # existing global calibration (unchanged, reused as-is)
        p, cal_global, yy, best_c_global = calibrate_full_history(probs, y, mask)
        global_full = np.full(T, np.nan)
        global_full[mask] = -np.log(np.clip(cal_global[np.arange(len(yy)), yy], EPS, 1.0))

        # new per-season calibration
        cal_season, temps = calibrate_per_season(probs, y, mask, seasons, season_order)
        season_full = -np.log(np.clip(cal_season[idx, y], EPS, 1.0))
        season_full = np.where(mask, season_full, np.nan)

        raw_ll[name], global_ll[name], season_ll[name] = raw_full, global_full, season_full

        valid = ~np.isnan(raw_full)
        rows.append({
            "name": name, "rounds": int(valid.sum()),
            "raw_log_loss": np.nanmean(raw_full), "global_cal_log_loss": np.nanmean(global_full),
            "per_season_cal_log_loss": np.nanmean(season_full),
        })
        for s, tval in temps.items():
            temp_rows.append({"series": name, "season": s, "T": tval})

    table = pd.DataFrame(rows).sort_values("per_season_cal_log_loss")
    table_path = os.path.join(OUT_DIR, "calibration_per_season_table.csv")
    table.to_csv(table_path, index=False)
    pd.set_option("display.width", 160)
    print(f"\n=== Raw vs global vs per-season calibration, full history — saved to {table_path} ===\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    for name in entries:
        for label, a, b in [
            ("per-season vs raw", season_ll[name], raw_ll[name]),
            ("per-season vs global", season_ll[name], global_ll[name]),
        ]:
            d, n = paired_diff(a, b)
            res = moving_block_bootstrap_test(d)
            sig_rows.append({"series": name, "comparison": label, "n": n, **res})

    sig_table = pd.DataFrame(sig_rows)
    sig_path = os.path.join(OUT_DIR, "calibration_per_season_significance.csv")
    sig_table.to_csv(sig_path, index=False)
    print(f"\n=== Significance (negative mean_diff = per-season is better) — saved to {sig_path} ===\n")
    print(sig_table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # plot restricted to the 4 algorithms + one reference bookmaker (B365C) --
    # full per-series detail (all Tier-A bookmakers) is in temp_rows/the CSV,
    # but 11 lines on one chart would be unreadable
    temps_table = pd.DataFrame(temp_rows)
    plot_series = list(algo_weights) + (["B365C"] if "B365C" in entries else [])
    plot_temperatures(temps_table[temps_table["series"].isin(plot_series)], season_order,
                       os.path.join(OUT_DIR, "calibration_per_season.png"))


if __name__ == "__main__":
    main()
