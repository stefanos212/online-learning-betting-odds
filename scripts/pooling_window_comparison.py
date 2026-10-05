"""
POOLING WINDOW COMPARISON: does the Tier-A ranking / significance story
change depending on how much history we pool?

Motivated by temporal_trends.py: the market's overround/margin genuinely
drifts over the decade (significantly, per the Mann-Kendall test there), so
pooling all 10 years into one comparison implicitly averages across seasons
that may not be directly comparable. This file reruns the SAME Tier-A
ranking pipeline (final_ranking.py) and a similar key-significance battery
(significance_test.py) independently on two separate windows:

  - RECENT_5YR: 2021/22-2025/26 (the original scope of this project, before
    the 10-year/22-league expansion -- now rerun on the full expanded
    22-league bookmaker panel, so it isn't a like-for-like repeat of the old
    10-league numbers, just the same TIME window)
  - FULL_10YR:  2016/17-2025/26 (everything)

Each window is refit FROM SCRATCH (a fresh uniform-weight start at the
beginning of that window), not sliced out of one continuous 10-year run --
this answers "what would we have concluded if this was all the data we had",
which is the actual question being asked.

Produces:
  - results/pooling_window_ranking_table.csv
  - results/pooling_window_significance_table.csv
  - results/pooling_window_comparison.png (slope chart: calibrated log-loss,
    5yr window vs 10yr window, for every entry present in both)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    OUT_DIR,
    load_full_universe, per_round_expert_loss,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping, evaluate,
)
from final_ranking import calibrate_full_history
from significance_test import moving_block_bootstrap_test, full_length_raw_logloss, full_length_calibrated_logloss, paired_diff

COVERAGE_THRESHOLD = 60.0  # keep in step with final_ranking.COVERAGE_THRESHOLD
WINDOWS = {
    "5yr": {"2122", "2223", "2324", "2425", "2526"},
    "10yr": None,  # None = every season
}
RANK_COLORS = {"algorithm": "#2a78d6", "bookmaker": "#eb6834"}


def run_window(P, awake, y, seasons, panel, season_set):
    mask = np.ones(len(y), dtype=bool) if season_set is None else np.isin(seasons, list(season_set))
    Pw, aw, yw = P[mask], awake[mask], y[mask]
    Tw, N, _ = Pw.shape
    loss = per_round_expert_loss(Pw, yw)

    W_hedge = run_hedge_sleeping(loss, aw, N)
    W_ogd = run_ogd_sleeping(loss, aw, N)
    W_ftrl = run_ftrl_sleeping(loss, aw, N)
    W_uniform = aw / aw.sum(axis=1, keepdims=True)
    algo_weights = {"Hedge": W_hedge, "OGD": W_ogd, "FTRL": W_ftrl, "Uniform average": W_uniform}

    coverage_pct = aw.mean(axis=0) * 100
    entries = {}
    for name, W in algo_weights.items():
        phat = np.einsum("tn,tnk->tk", W, Pw)
        entries[name] = (phat, np.ones(Tw, dtype=bool), "algorithm")
    for k, book in enumerate(panel):
        if coverage_pct[k] >= COVERAGE_THRESHOLD:
            entries[book] = (Pw[:, k, :], aw[:, k], "bookmaker")

    rows, raw_ll_full, cal_ll_full = [], {}, {}
    for name, (probs, m, kind) in entries.items():
        p, calibrated, yy, best_c = calibrate_full_history(probs, yw, m)
        raw_m = evaluate(p, yy)
        cal_m = evaluate(calibrated, yy)
        rows.append({
            "name": name, "type": kind, "coverage_pct": m.mean() * 100, "rounds": int(m.sum()),
            "raw_log_loss": raw_m["log_loss"], "calibrated_log_loss": cal_m["log_loss"],
        })
        raw_ll_full[name] = full_length_raw_logloss(probs, yw, Tw, mask=m)
        cal_ll_full[name] = full_length_calibrated_logloss(probs, yw, m, Tw)

    table = pd.DataFrame(rows).sort_values("calibrated_log_loss").reset_index(drop=True)

    comparisons = []
    if "PSC" in raw_ll_full:
        comparisons += [
            ("OGD vs PSC (raw)", raw_ll_full["OGD"], raw_ll_full["PSC"]),
            ("OGD vs PSC (calibrated)", cal_ll_full["OGD"], cal_ll_full["PSC"]),
        ]
    comparisons += [
        ("OGD vs Hedge (raw)", raw_ll_full["OGD"], raw_ll_full["Hedge"]),
        ("OGD vs Uniform average (raw)", raw_ll_full["OGD"], raw_ll_full["Uniform average"]),
        ("Hedge vs Uniform average (raw)", raw_ll_full["Hedge"], raw_ll_full["Uniform average"]),
    ]
    sig_rows = []
    for label, a, b in comparisons:
        d, n = paired_diff(a, b)
        res = moving_block_bootstrap_test(d)
        sig_rows.append({"comparison": label, "n": n, **res})

    return table, pd.DataFrame(sig_rows), Tw


def plot_comparison(ranking_table, out_path):
    piv = ranking_table.pivot_table(index=["name", "type"], columns="window", values="calibrated_log_loss")
    piv = piv.dropna().reset_index()  # only entries present (Tier A) in BOTH windows
    piv = piv.sort_values("10yr")

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(piv) + 2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ypos = np.arange(len(piv))[::-1]
    for yp, (_, row) in zip(ypos, piv.iterrows()):
        color = RANK_COLORS[row["type"]]
        ax.plot([row["5yr"], row["10yr"]], [yp, yp], color=color, linewidth=1.6, alpha=0.9, zorder=2)
        ax.scatter([row["5yr"]], [yp], s=70, facecolor="#fcfcfb", edgecolor=color, linewidth=1.8, zorder=3)
        ax.scatter([row["10yr"]], [yp], s=70, facecolor=color, edgecolor=color, linewidth=1.0, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(piv["name"], fontsize=9, color="#0b0b0b")
    ax.set_xlabel("Calibrated log-loss", color="#52514e")
    ax.set_title("5-year window (open circle) vs 10-year window (filled circle)\n"
                  "Tier-A entries present in both windows", color="#0b0b0b", fontsize=12, pad=12)
    ax.grid(True, axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", left=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def main():
    panel, P, awake, y, dates, seasons = load_full_universe()

    all_ranking_rows, all_sig_rows = [], []
    for window_name, season_set in WINDOWS.items():
        table, sig_table, Tw = run_window(P, awake, y, seasons, panel, season_set)
        table["window"] = window_name
        sig_table["window"] = window_name
        all_ranking_rows.append(table)
        all_sig_rows.append(sig_table)
        print(f"\n=== Window: {window_name} ({Tw} rounds) ===\n")
        pd.set_option("display.width", 140)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print()
        print(sig_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ranking_table = pd.concat(all_ranking_rows, ignore_index=True)
    sig_table = pd.concat(all_sig_rows, ignore_index=True)

    ranking_path = os.path.join(OUT_DIR, "pooling_window_ranking_table.csv")
    sig_path = os.path.join(OUT_DIR, "pooling_window_significance_table.csv")
    ranking_table.to_csv(ranking_path, index=False)
    sig_table.to_csv(sig_path, index=False)
    print(f"\nSaved: {ranking_path}")
    print(f"Saved: {sig_path}")

    plot_comparison(ranking_table, os.path.join(OUT_DIR, "pooling_window_comparison.png"))


if __name__ == "__main__":
    main()
