"""
Does de-vigging with Shin's method instead of basic proportional
normalization change how well the mixture algorithms do, INCLUDING relative
to individual bookmakers? Reruns the core "authoritative ranking" pipeline
(sleeping_experts.py's Hedge/OGD/FTRL/Uniform average + final_ranking.py's
Tier-A bookmaker selection and full-history calibration) independently on
odds_long.csv (basic normalization) and odds_long_shin.csv (Shin
normalization, see process_data_shin.py), then bootstrap-tests the
per-entity log-loss AND Brier score difference between the two (raw and
calibrated). Calibration QUALITY itself (ECE / reliability) is a separate
question, covered by shin_vs_basic_calibration.py.

This matters because results_table.csv / final_ranking_table.csv already
show that under basic normalization, PSC (Pinnacle closing odds) alone
beats every mixture algorithm -- the mixtures beat the uniform average and
every OTHER bookmaker, but not the single sharpest one. This script checks
whether that changes under Shin: does Shin normalization improve PSC's own
number too (it's de-vigged the same way as every other entry here), and do
the algorithms close the gap to it or not.

Scope deliberately stops at the core algorithm + Tier-A-bookmaker
comparison (raw + calibrated log-loss/Brier/RPS/accuracy, matching
final_ranking.py's methodology) rather than re-running every extension
script -- fixed_share.py, value_betting.py, contextual_experts.py etc. each
ask a question orthogonal to the de-vig method choice.

Both datasets are built from the identical match/bookmaker universe (the
de-vig formula can't change which matches or bookmakers have usable odds,
only their quoted probabilities), so the two panels line up round-for-round
in the same chronological order and the same bookmakers clear the Tier-A
coverage threshold in both -- asserted below before any comparison is made.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sleeping_experts import (
    DATA_DIR, OUT_DIR, EPS, OUTCOME_INDEX,
    per_round_expert_loss, run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping,
    evaluate,
)
from final_ranking import calibrate_full_history, COVERAGE_THRESHOLD, RANK_COLORS
from significance_test import moving_block_bootstrap_test, paired_diff

ALGORITHMS = ["Hedge", "OGD", "FTRL", "Uniform average"]
METRICS = ["log_loss", "brier"]  # bootstrap-tested per entity, raw and calibrated


def load_universe_from(csv_path):
    """Same construction as sleeping_experts.load_full_universe(), but reads
    an arbitrary long-format odds CSV instead of the hardcoded
    DATA_DIR/odds_long.csv -- lets this script build the (rounds x experts x
    outcomes) tensor from either the basic- or Shin-normalized dataset with
    identical logic, so the only difference between the two runs is the
    de-vig method."""
    long = pd.read_csv(csv_path, parse_dates=["Date"])
    panel = sorted(long["Bookmaker"].unique().tolist())
    N = len(panel)

    pivot = long.pivot(index="MatchID", columns="Bookmaker", values=["pH", "pD", "pA"])
    meta_all = long.drop_duplicates("MatchID").set_index("MatchID")[["Date", "FTR", "Season"]]

    awake_any = pivot["pH"].notna().any(axis=1)
    match_ids = pivot.index[awake_any]
    pivot = pivot.loc[match_ids]
    meta = meta_all.loc[match_ids].sort_values("Date", kind="mergesort")
    pivot = pivot.loc[meta.index]

    T = len(meta)
    P = np.full((T, N, 3), np.nan)
    for k, book in enumerate(panel):
        P[:, k, 0] = pivot[("pH", book)].values
        P[:, k, 1] = pivot[("pD", book)].values
        P[:, k, 2] = pivot[("pA", book)].values

    awake = ~np.isnan(P[:, :, 0])
    P = np.clip(np.nan_to_num(P, nan=EPS), EPS, 1.0)

    y = meta["FTR"].map(OUTCOME_INDEX).values
    dates = meta["Date"].values
    seasons = meta["Season"].astype(str).values
    return panel, P, awake, y, dates, seasons


def build_entries(panel, P, awake, y):
    """Mirrors final_ranking.py's entries construction: the 4 algorithm
    mixtures (always awake) plus every Tier-A bookmaker (coverage_pct >=
    COVERAGE_THRESHOLD), each as {name: (probs, mask, kind)}."""
    T, N, _ = P.shape
    loss = per_round_expert_loss(P, y)
    algo_weights = {
        "Hedge": run_hedge_sleeping(loss, awake, N),
        "OGD": run_ogd_sleeping(loss, awake, N),
        "FTRL": run_ftrl_sleeping(loss, awake, N),
        "Uniform average": awake / awake.sum(axis=1, keepdims=True),
    }
    entries = {}
    for name, W in algo_weights.items():
        phat = np.einsum("tn,tnk->tk", W, P)
        entries[name] = (phat, np.ones(T, dtype=bool), "algorithm")

    coverage_pct = awake.mean(axis=0) * 100
    for k, book in enumerate(panel):
        if coverage_pct[k] >= COVERAGE_THRESHOLD:
            entries[book] = (P[:, k, :], awake[:, k], "bookmaker")
    return entries


def per_round_log_loss(probs, y, idx):
    return -np.log(np.clip(probs[idx, y], EPS, 1.0))


def per_round_brier(probs, y, idx):
    onehot = np.zeros_like(probs)
    onehot[idx, y] = 1.0
    return ((probs - onehot) ** 2).sum(axis=1)


def full_length_metrics(probs, y, mask, T):
    """Per-round log-loss AND Brier, full T-length (NaN where `mask` is
    False), for both the raw series and its full-history calibrated
    version -- one calibrate_full_history() call serves both metrics
    (mirrors significance_test.py's full_length_raw/calibrated_logloss, but
    computes Brier alongside log-loss instead of needing a second pass, and
    returns the mask-restricted (p, calibrated, yy) so the caller can also
    get overall raw/calibrated evaluate() metrics without recalibrating)."""
    idx_T = np.arange(T)
    raw_ll = np.where(mask, per_round_log_loss(probs, y, idx_T), np.nan)
    raw_br = np.where(mask, per_round_brier(probs, y, idx_T), np.nan)

    p, calibrated, yy, best_c = calibrate_full_history(probs, y, mask)
    idx_m = np.arange(len(yy))
    cal_ll = np.full(T, np.nan)
    cal_ll[mask] = per_round_log_loss(calibrated, yy, idx_m)
    cal_br = np.full(T, np.nan)
    cal_br[mask] = per_round_brier(calibrated, yy, idx_m)

    return best_c, p, calibrated, yy, raw_ll, cal_ll, raw_br, cal_br


def plot_comparison(table, out_path):
    """Saves shin_vs_basic_comparison.png: one row per entity (4 algorithms
    + every Tier-A bookmaker), an open circle at the basic-normalization
    calibrated log-loss and a filled circle at the Shin-normalization one,
    connected by a line -- same slope-chart convention as
    final_ranking.plot_slope() (dot/slope, not a bar chart: these log-losses
    cluster near 1.0 with no meaningful zero), colored by type (algorithm
    vs. bookmaker) with the project's established RANK_COLORS
    (final_ranking.py) instead of inventing new colors."""
    basic = table[table["normalization"] == "Basic"].set_index("name")
    shin = table[table["normalization"] == "Shin"].set_index("name")
    order = basic.sort_values("calibrated_log_loss", ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(order) + 2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ypos = np.arange(len(order))
    for yp, name in zip(ypos, order):
        color = RANK_COLORS[basic.loc[name, "type"]]
        b = basic.loc[name, "calibrated_log_loss"]
        s = shin.loc[name, "calibrated_log_loss"]
        ax.plot([b, s], [yp, yp], color=color, linewidth=1.6, zorder=2, alpha=0.9)
        ax.scatter([b], [yp], s=70, facecolor="#fcfcfb", edgecolor=color, linewidth=1.8, zorder=3)
        ax.scatter([s], [yp], s=70, facecolor=color, edgecolor=color, linewidth=1.0, zorder=3)

    ax.set_yticks(ypos)
    ax.set_yticklabels(order, fontsize=8.5, color="#0b0b0b")
    ax.set_xlabel("Calibrated log-loss (full history)", color="#52514e")
    ax.set_title("Basic vs. Shin normalization: calibrated log-loss, algorithms + Tier-A bookmakers",
                  color="#0b0b0b", fontsize=12, pad=12)
    ax.grid(True, axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", left=False)

    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="#52514e", markersize=8, markeredgewidth=1.8, label="basic normalization"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#52514e",
               markeredgecolor="#52514e", markersize=8, label="Shin normalization"),
        Line2D([0], [0], color=RANK_COLORS["algorithm"], linewidth=2.5, label="algorithm"),
        Line2D([0], [0], color=RANK_COLORS["bookmaker"], linewidth=2.5, label="bookmaker"),
    ]
    legend = ax.legend(handles=legend_elements, frameon=False, loc="lower right", fontsize=8)
    for text in legend.get_texts():
        text.set_color("#0b0b0b")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def main():
    """Loads both datasets, asserts they share the same round universe,
    builds algorithm + Tier-A-bookmaker entries on each, evaluates raw +
    full-history-calibrated, saves the combined table, bootstrap-tests Shin
    vs. Basic per entity (raw and calibrated), and saves the comparison
    plot."""
    paths = {"Basic": os.path.join(DATA_DIR, "odds_long.csv"),
             "Shin": os.path.join(DATA_DIR, "odds_long_shin.csv")}

    datasets = {}
    for label, path in paths.items():
        panel, P, awake, y, dates, seasons = load_universe_from(path)
        T = P.shape[0]
        print(f"{label:5s}: {P.shape[1]} experts, {T} rounds "
              f"({pd.Timestamp(dates.min()).date()} -> {pd.Timestamp(dates.max()).date()})")
        datasets[label] = {"panel": panel, "P": P, "awake": awake, "y": y, "T": T}

    # de-vig method can't change which matches/bookmakers have usable odds,
    # so both panels must line up round-for-round before any comparison
    assert datasets["Basic"]["T"] == datasets["Shin"]["T"], \
        "Basic and Shin datasets disagree on round count -- can't align for comparison"
    assert np.array_equal(datasets["Basic"]["y"], datasets["Shin"]["y"]), \
        "Basic and Shin datasets disagree on match outcomes -- can't align for comparison"
    T = datasets["Basic"]["T"]
    y = datasets["Basic"]["y"]

    rows = []
    per_round = {}  # (label, name, metric, "raw"/"calibrated") -> full-length array
    for label, ds in datasets.items():
        entries = build_entries(ds["panel"], ds["P"], ds["awake"], y)
        for name, (probs, mask, kind) in entries.items():
            best_c, p, calibrated, yy, raw_ll, cal_ll, raw_br, cal_br = full_length_metrics(probs, y, mask, T)
            per_round[(label, name, "log_loss", "raw")] = raw_ll
            per_round[(label, name, "log_loss", "calibrated")] = cal_ll
            per_round[(label, name, "brier", "raw")] = raw_br
            per_round[(label, name, "brier", "calibrated")] = cal_br

            raw_m = evaluate(p, yy)
            cal_m = evaluate(calibrated, yy)

            rows.append({
                "name": name, "type": kind, "normalization": label,
                "coverage_pct": mask.mean() * 100, "best_lr_scale": best_c,
                "raw_log_loss": raw_m["log_loss"], "calibrated_log_loss": cal_m["log_loss"],
                "raw_brier": raw_m["brier"], "calibrated_brier": cal_m["brier"],
                "raw_rps": raw_m["rps"], "calibrated_rps": cal_m["rps"],
                "raw_accuracy": raw_m["accuracy"], "calibrated_accuracy": cal_m["accuracy"],
            })
            print(f"  {label:5s} {kind:9s} {name:16s} raw_log_loss={raw_m['log_loss']:.5f}  "
                  f"calibrated_log_loss={cal_m['log_loss']:.5f}  raw_brier={raw_m['brier']:.5f}  "
                  f"calibrated_brier={cal_m['brier']:.5f}")

    table = pd.DataFrame(rows).sort_values(["normalization", "calibrated_log_loss"]).reset_index(drop=True)
    table_path = os.path.join(OUT_DIR, "shin_vs_basic_table.csv")
    table.to_csv(table_path, index=False)

    pd.set_option("display.width", 160)
    for label in ["Basic", "Shin"]:
        sub = table[table["normalization"] == label].sort_values("calibrated_log_loss")
        print(f"\n=== {label} normalization -- ranked by calibrated log-loss ===")
        print(sub[["name", "type", "coverage_pct", "raw_log_loss", "calibrated_log_loss"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved: {table_path}")

    # ---- significance: Shin vs Basic, per entity, per metric, raw and calibrated ----
    entity_names = [n for n in table["name"].unique()]
    sig_rows = []
    for name in entity_names:
        kind = table.loc[table["name"] == name, "type"].iloc[0]
        for metric in METRICS:
            for stage in ["raw", "calibrated"]:
                a = per_round[("Shin", name, metric, stage)]
                b = per_round[("Basic", name, metric, stage)]
                diff, n = paired_diff(a, b)
                res = moving_block_bootstrap_test(diff)
                sig_rows.append({"name": name, "type": kind, "metric": metric, "stage": stage, **res})

    sig_table = pd.DataFrame(sig_rows)

    # ---- does calibration still help WITHIN each normalization? i.e. is
    # temperature scaling redundant once Shin has already partly corrected
    # the favorite-longshot bias, or does it still add something? Same
    # question significance_test.py asks for Basic ("X: raw vs calibrated"),
    # repeated here for Shin so the two are directly comparable. ----
    cal_rows = []
    for label in ["Basic", "Shin"]:
        for name in entity_names:
            kind = table.loc[table["name"] == name, "type"].iloc[0]
            for metric in METRICS:
                raw = per_round[(label, name, metric, "raw")]
                cal = per_round[(label, name, metric, "calibrated")]
                diff, n = paired_diff(cal, raw)  # negative = calibration HELPS (lower loss)
                res = moving_block_bootstrap_test(diff)
                cal_rows.append({"name": name, "type": kind, "normalization": label, "metric": metric, **res})
    cal_table = pd.DataFrame(cal_rows)

    sig_path = os.path.join(OUT_DIR, "shin_vs_basic_significance.csv")
    sig_table.to_csv(sig_path, index=False)
    cal_path = os.path.join(OUT_DIR, "shin_vs_basic_calibration_benefit.csv")
    cal_table.to_csv(cal_path, index=False)

    for metric in METRICS:
        print(f"\n=== Moving block bootstrap: Shin vs. Basic per entity, {metric} "
              f"(negative mean_diff = Shin has the LOWER/better {metric}) ===\n")
        sub = sig_table[sig_table["metric"] == metric].sort_values(["type", "name", "stage"])
        for _, r in sub.iterrows():
            flag = "SIGNIFICANT" if r["significant_95"] else "not significant"
            print(f"{r['type']:9s} {r['name']:16s} {r['stage']:11s} n={int(r['n']):6d}  "
                  f"mean_diff={r['mean_diff']:+.5f}  95% CI=[{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]  "
                  f"p={r['p_value']:.4f}  {flag}")
    print(f"\nSaved: {sig_path}")

    for metric in METRICS:
        print(f"\n=== Moving block bootstrap: does calibration still help WITHIN each normalization, {metric} "
              f"(negative mean_diff = calibration HELPS) ===\n")
        sub = cal_table[cal_table["metric"] == metric].sort_values(["type", "name", "normalization"])
        for _, r in sub.iterrows():
            flag = "SIGNIFICANT" if r["significant_95"] else "not significant"
            print(f"{r['normalization']:5s} {r['type']:9s} {r['name']:16s} n={int(r['n']):6d}  "
                  f"mean_diff={r['mean_diff']:+.5f}  95% CI=[{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]  "
                  f"p={r['p_value']:.4f}  {flag}")
    print(f"\nSaved: {cal_path}")

    plot_comparison(table, os.path.join(OUT_DIR, "shin_vs_basic_comparison.png"))


if __name__ == "__main__":
    main()
