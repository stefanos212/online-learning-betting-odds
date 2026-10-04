"""
WARM-START per-season temperature calibration -- a third point on the
spectrum between calibration_correction.py's GLOBAL fit and
calibration_per_season.py's (cold-start) PER-SEASON fit.

Recap of the trade-off calibration_per_season.py identified:
  - GLOBAL (final_ranking.calibrate_full_history): one continuous causal OGD
    fit over the full 10-year history. eta_t = c/sqrt(t+1) shrinks over the
    whole decade, so by the later seasons the fitted s is essentially frozen
    near a long-run compromise -- it can't track a genuinely time-varying
    bias, even though temporal_trends.py found one exists season to season.
  - PER-SEASON (calibration_per_season.calibrate_per_season): resets BOTH the
    round counter t (so eta_t is fast/responsive again) AND the temperature
    itself (s back to 1.0, "no correction") at the start of every season.
    That fixes the tracking problem but pays a cold-start cost 10 times over,
    on a much smaller per-season sample (~7,658 rounds vs 76,584) -- found to
    be significantly better than raw, but statistically indistinguishable
    from global overall, plausibly because the repeated cold-start cost and
    smaller sample cancel out the benefit of tracking drift.

WARM-START (this file): reset only the round counter t at each season
boundary (so the learning rate is fast/responsive within a season, exactly
like calibrate_per_season) but carry the PREVIOUS season's final fitted s
forward as this season's starting point, instead of resetting to s=1.0. The
first season still starts cold at s=1.0 (there is no prior season to inherit
from). This should remove the repeated cold-start penalty while keeping the
ability to track within-season drift -- whether that actually beats both
existing variants, rather than landing between them, is an empirical
question this file measures directly, on equal footing (same full-history
evaluation protocol as final_ranking.py / calibration_per_season.py).

temperature_calibrate_warmstart() / select_lr_on_train_warmstart() are small
copies of calibration_correction.py's temperature_calibrate() /
select_lr_on_train() with one added parameter (the starting s) -- duplicated
rather than added as an optional arg to the originals, so
calibration_correction.py (which final_ranking.py, significance_test.py,
value_betting.py, contextual_experts.py and calibration_per_season.py all
already import from) never has to change.

Produces:
  - results/calibration_warmstart_table.csv          (raw vs global vs
                                                        per-season vs
                                                        warm-start, full
                                                        history, for the 4
                                                        algorithms + Tier-A
                                                        bookmakers)
  - results/calibration_warmstart_significance.csv    (bootstrap tests:
                                                        warm-start vs raw,
                                                        vs global, vs
                                                        cold-start per-season)
  - results/calibration_warmstart.png                 (per-season fitted T,
                                                        cold-start reset
                                                        (dashed) vs
                                                        warm-start (solid) --
                                                        direct visual evidence
                                                        of whether warm-start
                                                        removes the sawtooth)
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
from calibration_correction import TRAIN_FRACTION, LR_GRID, S_MIN, S_MAX
from calibration_per_season import calibrate_per_season
from final_ranking import COVERAGE_THRESHOLD
from significance_test import (
    moving_block_bootstrap_test, paired_diff,
    full_length_raw_logloss, full_length_calibrated_logloss,
)

SERIES_COLORS = {**COLORS, "B365C": COLORS["benchmark"]}


# ------------------------------------------------- warm-start calibrator ---

def temperature_calibrate_warmstart(probs, y, lr_scale, s_init):
    """Identical recipe to calibration_correction.temperature_calibrate(),
    except s starts at `s_init` instead of the hardcoded 1.0 -- the one
    change needed to carry a prior season's fitted temperature forward."""
    T = len(y)
    log_phat = np.log(np.clip(probs, EPS, 1.0))
    s = s_init
    calibrated = np.zeros((T, 3))
    s_hist = np.zeros(T)
    for t in range(T):
        s_hist[t] = s
        exps = np.exp(s * log_phat[t])
        w = exps / exps.sum()
        calibrated[t] = w
        grad = -log_phat[t, y[t]] + np.dot(w, log_phat[t])
        eta = lr_scale / np.sqrt(t + 1)
        s = np.clip(s - eta * grad, S_MIN, S_MAX)
    return calibrated, s_hist


def select_lr_on_train_warmstart(probs_train, y_train, lr_grid, s_init):
    """Same train-only learning-rate selection as select_lr_on_train(), but
    every candidate rate is evaluated starting from s_init instead of 1.0, so
    the chosen rate is appropriate for a warm-started fit."""
    best_c, best_loss = None, np.inf
    idx = np.arange(len(y_train))
    for c in lr_grid:
        calibrated, _ = temperature_calibrate_warmstart(probs_train, y_train, lr_scale=c, s_init=s_init)
        loss = -np.log(np.clip(calibrated[idx, y_train], EPS, 1.0)).mean()
        if loss < best_loss:
            best_loss, best_c = loss, c
    return best_c, best_loss


def calibrate_per_season_warmstart(probs, y, mask, seasons, season_order):
    """Same per-season loop as calibration_per_season.calibrate_per_season():
    round counter t resets to 0 at every season boundary. Difference: s is
    carried over from the previous season's final fitted value instead of
    being reset to 1.0 (first season still starts cold, at s=1.0, since there
    is no prior season). Returns a full-length calibrated array (aligned to
    the original T-length index, NaN where masked out) plus a per-season dict
    of the fitted final temperature T=1/s."""
    T = len(y)
    calibrated_full = np.full((T, 3), np.nan)
    final_T_by_season = {}
    s_carry = 1.0

    for s in season_order:
        sel = mask & (seasons == s)
        if not sel.any():
            continue
        p, yy = probs[sel], y[sel]
        n = len(yy)
        n_train = max(1, int(round(n * TRAIN_FRACTION)))
        best_c, _ = select_lr_on_train_warmstart(p[:n_train], yy[:n_train], LR_GRID, s_carry)
        calibrated, s_hist = temperature_calibrate_warmstart(p, yy, lr_scale=best_c, s_init=s_carry)
        calibrated_full[sel] = calibrated
        final_T_by_season[s] = 1.0 / s_hist[-1]
        s_carry = s_hist[-1]

    return calibrated_full, final_T_by_season


def _calibrated_log_loss(calibrated, y, mask, idx):
    """log-loss from an already-computed (T,3) calibrated array that is
    already NaN-scattered outside `mask` -- the return contract shared by
    calibrate_per_season() and calibrate_per_season_warmstart()."""
    ll = -np.log(np.clip(calibrated[idx, y], EPS, 1.0))
    return np.where(mask, ll, np.nan)


# --------------------------------------------------------------- plotting --

def plot_temperature_comparison(temps_table, season_order, out_path):
    """One pair of lines per series: dashed = cold-start per-season (T reset
    to 1.0 every season boundary), solid = warm-start (T carried over).
    Directly shows whether warm-starting removes the "snap back toward T=1"
    sawtooth of the cold-start method."""
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axhline(1.0, color="#898781", linewidth=1.2, linestyle="--", zorder=1, label="T=1 (no correction)")

    xpos = np.arange(len(season_order))
    for series in temps_table["series"].unique():
        color = SERIES_COLORS.get(series, "#898781")
        for method, ls, alpha in [("cold-start reset", "--", 0.5), ("warm-start", "-", 1.0)]:
            sub = temps_table[(temps_table["series"] == series) & (temps_table["method"] == method)]
            sub = sub.set_index("season").reindex(season_order)
            ax.plot(xpos, sub["T"], color=color, linewidth=1.8, alpha=alpha, linestyle=ls,
                     marker="o", markersize=4, label=f"{series} ({method})", zorder=3)

    ax.set_xticks(xpos)
    ax.set_xticklabels(season_order, rotation=45, fontsize=8)
    ax.set_ylabel("Fitted temperature T (per season)", color="#52514e")
    ax.set_title("Per-season fitted temperature: cold-start reset (dashed) vs. warm-start (solid)",
                  color="#0b0b0b", fontsize=12, pad=12)
    _style_axes(ax)
    legend = ax.legend(frameon=False, loc="best", fontsize=6.5, ncol=2)
    for text in legend.get_texts():
        text.set_color("#0b0b0b")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


# ---------------------------------------------------------------- main -----

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
    idx = np.arange(T)

    for name, (probs, mask) in entries.items():
        print(f"  processing {name}...")

        raw_full = full_length_raw_logloss(probs, y, T, mask=mask)
        global_full = full_length_calibrated_logloss(probs, y, mask, T)

        cal_season, temps_season = calibrate_per_season(probs, y, mask, seasons, season_order)
        season_full = _calibrated_log_loss(cal_season, y, mask, idx)

        cal_warm, temps_warm = calibrate_per_season_warmstart(probs, y, mask, seasons, season_order)
        warm_full = _calibrated_log_loss(cal_warm, y, mask, idx)

        valid = ~np.isnan(raw_full)
        rows.append({
            "name": name, "rounds": int(valid.sum()),
            "raw_log_loss": np.nanmean(raw_full),
            "global_cal_log_loss": np.nanmean(global_full),
            "per_season_cal_log_loss": np.nanmean(season_full),
            "warmstart_cal_log_loss": np.nanmean(warm_full),
        })
        for s, tval in temps_season.items():
            temp_rows.append({"series": name, "season": s, "method": "cold-start reset", "T": tval})
        for s, tval in temps_warm.items():
            temp_rows.append({"series": name, "season": s, "method": "warm-start", "T": tval})

        for label, a, b in [
            ("warm-start vs raw", warm_full, raw_full),
            ("warm-start vs global", warm_full, global_full),
            ("warm-start vs per-season (cold-start reset)", warm_full, season_full),
        ]:
            d, n = paired_diff(a, b)
            res = moving_block_bootstrap_test(d)
            sig_rows.append({"series": name, "comparison": label, "n": n, **res})

    table = pd.DataFrame(rows).sort_values("warmstart_cal_log_loss")
    table_path = os.path.join(OUT_DIR, "calibration_warmstart_table.csv")
    table.to_csv(table_path, index=False)
    pd.set_option("display.width", 160)
    print(f"\n=== Raw vs global vs per-season vs warm-start calibration, full history — saved to {table_path} ===\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    sig_table = pd.DataFrame(sig_rows)
    sig_path = os.path.join(OUT_DIR, "calibration_warmstart_significance.csv")
    sig_table.to_csv(sig_path, index=False)
    print(f"\n=== Significance (negative mean_diff = warm-start is better) — saved to {sig_path} ===\n")
    print(sig_table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    temps_table = pd.DataFrame(temp_rows)
    plot_series = list(algo_weights)
    if "B365C" in entries:
        plot_series.append("B365C")
    plot_temperature_comparison(temps_table[temps_table["series"].isin(plot_series)], season_order,
                                 os.path.join(OUT_DIR, "calibration_warmstart.png"))


if __name__ == "__main__":
    main()
