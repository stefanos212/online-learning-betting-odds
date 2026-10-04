"""
STACKING THE WINS: WHAT IS THE BEST CONFIGURATION WE CAN ACTUALLY BUILD?

Four things have been measured separately in this project and never combined:
the corrected OGD step (applied), the closing-only panel beating the full one
(significant, 4/4 algorithms, both normalizations), Shin de-vig beating basic
on raw log-loss, and calibration. The main pipeline still runs on the FULL
panel with BASIC normalization, i.e. it leaves two of the four on the table.

This file runs the stack end to end and asks the one question that could
change the project's headline: does any configuration beat PSC, the best
single bookmaker, on a like-for-like comparison?

CONFIGURATIONS
  baseline        full 26-expert panel, basic de-vig      (what the pipeline ships)
  closing+basic   closing-only panel, basic de-vig        (isolates the panel change)
  full+Shin       full panel, Shin de-vig                 (isolates the de-vig change)
  A               closing-only panel, Shin de-vig         (both)
  B               A minus single-season bookmakers        (both, plus a quality filter)

All use OGD with the tuned step, and each is reported raw and causally
calibrated.

TWO THINGS THAT MAKE THIS AN HONEST COMPARISON, both of which this project
has been bitten by before:

  1. EVERY configuration is scored on the SAME rounds -- the intersection of
     every config's own coverage with PSC's -- because the panels have
     different coverage and a log-loss computed over a different match set is
     not comparable. Fitting still uses each config's full support; only the
     scoring is restricted.

  2. Choosing a configuration by looking at these numbers IS selection, so
     everything is reported twice: full sample (descriptive) and a held-out
     chronological tail the choice never saw. Quote the tail.

Note the mixture CONTAINS PSC in every closing configuration -- this is a
forecasting comparison, not a betting one, so that is legitimate, but it
means "we beat PSC" reads as "the blend beats its own best member", not as
"we beat it from outside". `no_pinnacle_panel.py` is the experiment that
removes it.

Produces:
  - results/best_configuration.csv
  - results/best_configuration_significance.csv
  - results/best_configuration.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    DATA_DIR, OUT_DIR, EPS, COLORS,
    per_round_expert_loss, run_ogd_sleeping,
    load_universe_from, split_by_market_phase,
)
from calibration_analysis import _style_axes
from final_ranking import calibrate_full_history
from significance_test import moving_block_bootstrap_test, paired_diff

TRAIN_FRACTION = 0.75
ANCHOR = "PSC"


def single_season(panel, awake, seasons):
    """Bookmakers active in exactly one season -- their apparent quality is
    confounded with that season's difficulty (see CLAUDE.md), and they are the
    ones that hold inflated frozen weights (see fixed_share.py)."""
    out = []
    for k, name in enumerate(panel):
        if len(set(seasons[awake[:, k]])) <= 1:
            out.append(name)
    return out


def fit_config(P, awake, y, indices):
    """OGD (tuned step) on a column subset, then causal calibration. Returns
    per-round raw and calibrated log-loss at FULL length, NaN where the subset
    has nobody awake."""
    T = P.shape[0]
    P_s, aw_s = P[:, indices, :], awake[:, indices]
    alive = aw_s.any(axis=1)

    W = run_ogd_sleeping(per_round_expert_loss(P_s, y), aw_s, len(indices))
    phat = np.einsum("tn,tnk->tk", W, P_s)

    raw = np.full(T, np.nan)
    raw[alive] = -np.log(np.clip(phat[alive][np.arange(int(alive.sum())), y[alive]], EPS, 1.0))

    _, cal, yy, _ = calibrate_full_history(phat, y, alive)
    calib = np.full(T, np.nan)
    calib[alive] = -np.log(np.clip(cal[np.arange(len(yy)), yy], EPS, 1.0))
    return raw, calib, int(alive.sum())


def plot_configs(table, out_path):
    cal = table[table["stage"] == "calibrated"].sort_values("val_log_loss")
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(cal) + 2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ypos = np.arange(len(cal))[::-1]
    psc = cal[cal["config"] == "PSC alone"]["val_log_loss"]
    if len(psc):
        ax.axvline(psc.iloc[0], color="#898781", linestyle="--", linewidth=1.4,
                    zorder=1, label="PSC (held-out)")
    for yp, (_, r) in zip(ypos, cal.iterrows()):
        color = "#898781" if r["config"] == "PSC alone" else COLORS["OGD"]
        ax.scatter([r["val_log_loss"]], [yp], s=80, color=color, zorder=3,
                    edgecolor="#fcfcfb", linewidth=1.1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(cal["config"], fontsize=9, color="#0b0b0b")
    ax.set_xlabel("Calibrated log-loss, held-out tail (same rounds for every row)",
                   color="#52514e")
    ax.set_title("Stacking the measured wins: does any configuration beat PSC?",
                  color="#0b0b0b", fontsize=12, pad=12)
    _style_axes(ax)
    ax.grid(False, axis="y")
    leg = ax.legend(frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color("#0b0b0b")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def main():
    basic = load_universe_from(os.path.join(DATA_DIR, "odds_long.csv"))
    shin = load_universe_from(os.path.join(DATA_DIR, "odds_long_shin.csv"))
    panel, P_b, awake, y, dates, seasons = basic
    _, P_s, awake_s, y_s, _, _ = shin
    assert np.array_equal(y, y_s) and np.array_equal(awake, awake_s), \
        "the two de-vig datasets must line up round-for-round"
    T, N, _ = P_b.shape

    open_idx, close_idx = split_by_market_phase(panel)
    ss = single_season(panel, awake, seasons)
    close_multi = [k for k in close_idx if panel[k] not in ss]
    print(f"{T} rounds, {N} experts")
    print(f"closing panel: {len(close_idx)} experts")
    print(f"single-season among them: {[panel[k] for k in close_idx if panel[k] in ss]}")
    print(f"closing, multi-season only: {len(close_multi)} experts "
          f"-- {[panel[k] for k in close_multi]}\n")

    configs = {
        "baseline (full 26, basic)":   (P_b, list(range(N))),
        "closing only, basic":         (P_b, close_idx),
        "full 26, Shin":               (P_s, list(range(N))),
        "A: closing + Shin":           (P_s, close_idx),
        "B: A minus single-season":    (P_s, close_multi),
    }

    series = {}
    for name, (P_use, idx) in configs.items():
        raw, cal, n_alive = fit_config(P_use, awake, y, idx)
        series[name] = {"raw": raw, "calibrated": cal}
        print(f"  fitted {name:28s} ({len(idx):2d} experts, {n_alive} rounds covered)")

    # PSC on its own, as the thing to beat
    k = panel.index(ANCHOR)
    psc_mask = awake[:, k]
    psc_raw = np.full(T, np.nan)
    psc_raw[psc_mask] = -np.log(np.clip(
        P_b[psc_mask][np.arange(int(psc_mask.sum())), k, y[psc_mask]], EPS, 1.0))
    _, psc_cal_arr, yy, _ = calibrate_full_history(P_b[:, k, :], y, psc_mask)
    psc_cal = np.full(T, np.nan)
    psc_cal[psc_mask] = -np.log(np.clip(psc_cal_arr[np.arange(len(yy)), yy], EPS, 1.0))
    series["PSC alone"] = {"raw": psc_raw, "calibrated": psc_cal}

    # common rounds: every configuration AND PSC defined
    common = np.ones(T, dtype=bool)
    for s in series.values():
        common &= ~np.isnan(s["calibrated"]) & ~np.isnan(s["raw"])
    n_common = int(common.sum())
    n_train = int(round(n_common * TRAIN_FRACTION))
    idx_common = np.flatnonzero(common)
    val = np.zeros(T, dtype=bool)
    val[idx_common[n_train:]] = True
    tr = np.zeros(T, dtype=bool)
    tr[idx_common[:n_train]] = True
    print(f"\nscoring every configuration on the SAME {n_common} rounds "
          f"(train {n_train} / held-out {n_common - n_train})\n")

    rows = []
    for name, s in series.items():
        for stage in ["raw", "calibrated"]:
            v = s[stage]
            rows.append({"config": name, "stage": stage,
                          "full_log_loss": float(np.nanmean(np.where(common, v, np.nan))),
                          "train_log_loss": float(np.nanmean(np.where(tr, v, np.nan))),
                          "val_log_loss": float(np.nanmean(np.where(val, v, np.nan)))})
    table = pd.DataFrame(rows)
    tp = os.path.join(OUT_DIR, "best_configuration.csv")
    table.to_csv(tp, index=False)

    pd.set_option("display.width", 200)
    print(f"=== all configurations, same rounds — saved to {tp} ===\n")
    print(table.sort_values(["stage", "val_log_loss"])
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # paired tests against PSC and against the baseline, on the common rounds
    sig_rows = []
    for name in series:
        if name == "PSC alone":
            continue
        for stage in ["raw", "calibrated"]:
            for ref in ["PSC alone", "baseline (full 26, basic)"]:
                if name == ref:
                    continue
                a = np.where(common, series[name][stage], np.nan)
                b = np.where(common, series[ref][stage], np.nan)
                d, n = paired_diff(a, b)
                res = moving_block_bootstrap_test(d)
                verdict = ("tie (not significant)" if not res["significant_95"]
                           else (f"{name} better" if res["mean_diff"] < 0 else f"{ref} better"))
                sig_rows.append({"config": name, "stage": stage, "vs": ref,
                                  "n": n, "verdict": verdict, **res})
    sig = pd.DataFrame(sig_rows)
    sp = os.path.join(OUT_DIR, "best_configuration_significance.csv")
    sig.to_csv(sp, index=False)

    print(f"\n=== paired tests on the common rounds — saved to {sp} ===")
    print("    (negative mean_diff = the configuration is better)\n")
    for ref in ["PSC alone", "baseline (full 26, basic)"]:
        sub = sig[(sig["vs"] == ref) & (sig["stage"] == "calibrated")]
        print(f"-- calibrated, vs {ref} --")
        print(sub[["config", "n", "mean_diff", "ci_low", "ci_high", "p_value", "verdict"]]
              .to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        print()

    # the decision
    cal = table[(table["stage"] == "calibrated") & (table["config"] != "PSC alone")]
    best_train = cal.sort_values("train_log_loss").iloc[0]
    best_val = cal.sort_values("val_log_loss").iloc[0]
    psc_val = table[(table["config"] == "PSC alone") &
                     (table["stage"] == "calibrated")]["val_log_loss"].iloc[0]
    print("=== decision ===")
    print(f"  chosen on TRAIN      : {best_train['config']}  (train {best_train['train_log_loss']:.5f})")
    print(f"  its held-out score   : {best_train['val_log_loss']:.5f}")
    print(f"  best on held-out     : {best_val['config']}  ({best_val['val_log_loss']:.5f})")
    print(f"  PSC on held-out      : {psc_val:.5f}")
    print(f"  -> train-chosen config beats PSC on held-out: "
          f"{best_train['val_log_loss'] < psc_val}")

    plot_configs(table, os.path.join(OUT_DIR, "best_configuration.png"))


if __name__ == "__main__":
    main()
