"""
MARKET STRUCTURE / DE-VIG METHOD comparisons -- every "does changing how the
data is built change the answer" experiment in the project, consolidated
into one file. This is distinct from value_betting.py: that file asks
whether the mixture is economically exploitable; this one asks whether the
mixture's SCORING (log-loss/Brier/calibration) depends on choices made
upstream of the algorithms -- which market snapshot (opening vs. closing
odds) and which de-vig formula (basic proportional vs. Shin) the panel is
built from.

Four experiments (`run_*`), in the order `main()` runs them:

  1. `run_opening_vs_closing` -- football-data.co.uk quotes two prices per
     bookmaker for most of the panel: an OPENING price (column prefix
     "{book}") and a CLOSING price ("{book}C"), taken just before kickoff
     after all the money has moved the line. data_processer.py's hardcoded
     prefix list preserves this "C" suffix verbatim (including through the
     VC/BV rebrand merge), so the 26-bookmaker panel splits cleanly into 13
     opening / 13 closing. Reruns Hedge/OGD/FTRL/Uniform average on three
     panels -- opening-only, closing-only, and the full 26 -- to see
     whether the market's final, most-informed price makes a better
     mixture than its first, and whether closing-only actually beats
     throwing every bookmaker (opening included) into one mixture, or just
     looks better than opening alone while still losing to the full panel.

  2. `run_quality_transfer_correlation` -- descriptive, no leakage risk: is
     a bookmaker's CLOSING-side skill predictive of its own OPENING-side
     skill? I.e. is sharpness a persistent property of the underlying
     odds-setter, or purely a function of how late the price was set?
     Pearson + Spearman correlation across the 13 opening/closing pairs.
     (Motivated by value_betting.py's leave-one-out design: excluding only
     a bookmaker's own column still leaves its closing counterpart and
     every other closing bookmaker in the training panel when betting
     against an OPENING target -- see that file's module docstring for how
     it corrects for this. This experiment doesn't correct anything itself,
     it just checks whether the underlying premise, "closing skill predicts
     opening skill", holds.)

  3. `run_shin_vs_basic_comparison` -- does de-vigging with Shin's method
     instead of basic proportional normalization change how well the
     mixture algorithms do, INCLUDING relative to individual bookmakers?
     Reruns the core "authoritative ranking" pipeline (sleeping_experts.py's
     4 algorithms + final_ranking.py's Tier-A bookmaker selection and
     full-history calibration) independently on odds_long.csv and
     odds_long_shin.csv, then bootstrap-tests the per-entity log-loss and
     Brier difference (raw and calibrated), plus whether calibration still
     helps within each normalization. Motivated by results_table.csv
     already showing that under basic normalization, PSC (Pinnacle
     closing) alone beats every mixture algorithm -- does Shin change that?

  4. `run_shin_vs_basic_calibration` -- does Shin normalization actually
     produce better-CALIBRATED raw probabilities, i.e. does it do what it's
     supposed to do? Experiment 3 answers "which gets the lower loss"; this
     one answers the more direct mechanistic question calibration_analysis.py
     originally asked of the basic panel: among all the times a series said
     "~30% probability", did the outcome happen ~30% of the time? Since
     Shin's method is specifically designed to correct the favorite-longshot
     bias, the expectation is a smaller ECE and a flatter calibration-gap
     curve under Shin -- checked here rather than assumed.

Produces:
  results/opening_vs_closing_table.csv, _significance.csv, .png,
    _ranking.csv, _ranking.png
  results/opening_closing_quality_transfer.csv, .png
  results/shin_vs_basic_table.csv, _significance.csv,
    _calibration_benefit.csv, .png
  results/shin_vs_basic_calibration_table.csv, _ece.csv,
    _reliability.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sleeping_experts import (
    DATA_DIR, OUT_DIR, EPS, COLORS, OUTCOME_INDEX, EXCLUDED_BOOKMAKERS,
    load_full_universe, load_universe_from, split_by_market_phase,
    per_round_expert_loss,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping, evaluate,
)
from final_ranking import calibrate_full_history, COVERAGE_THRESHOLD, RANK_COLORS
from significance_test import moving_block_bootstrap_test, paired_diff
from calibration_analysis import (
    REFERENCE_BOOKMAKERS, SERIES_COLORS,
    pooled_prob_outcome_pairs, calibration_table, expected_calibration_error, _style_axes,
)

ALGO_ORDER = ["Hedge", "OGD", "FTRL", "Uniform average"]
METRICS = ["log_loss", "brier"]  # bootstrap-tested per entity in experiment 3, raw and calibrated
SERIES_ORDER = ALGO_ORDER + REFERENCE_BOOKMAKERS  # experiment 4


# =====================================================================
# shared building blocks
# =====================================================================

def fit_all_algorithms(P, awake, y):
    """The 4 mixture algorithms (Hedge/OGD/FTRL/Uniform average) fitted on
    one panel, returns {name: phat (T, 3)}. This exact weight-computation
    block used to be copy-pasted three times across this file's four source
    scripts (opening_vs_closing.py's run_all_algorithms,
    shin_vs_basic_comparison.py's build_entries,
    shin_vs_basic_calibration.py's calibration_for_dataset)."""
    T, N, _ = P.shape
    loss = per_round_expert_loss(P, y)
    algo_weights = {
        "Hedge": run_hedge_sleeping(loss, awake, N),
        "OGD": run_ogd_sleeping(loss, awake, N),
        "FTRL": run_ftrl_sleeping(loss, awake, N),
        "Uniform average": awake / awake.sum(axis=1, keepdims=True),
    }
    return {name: np.einsum("tn,tnk->tk", W, P) for name, W in algo_weights.items()}


# =====================================================================
# 1. opening vs. closing odds (basic normalization only)
# =====================================================================

def build_subset(P, awake, y, indices):
    """Restrict the full panel to `indices` (the opening or closing subset),
    then drop rounds where NONE of those columns are awake -- the same
    reduction load_full_universe() applies at the full-panel level, re-
    applied to a column subset. Returns the reduced (P, awake, y) plus
    `keep`, the boolean mask (full T length) of which rounds survived, so
    per-round results can be scattered back for cross-subset comparison."""
    sub_awake_full = awake[:, indices]
    keep = sub_awake_full.any(axis=1)
    P_sub = P[:, indices, :][keep]
    return P_sub, sub_awake_full[keep], y[keep], keep


def plot_market_comparison(table, out_path, label="Basic"):
    """Saves results/opening_vs_closing.png: one row per algorithm, open
    circle = opening-odds mixture log-loss, filled circle = closing-odds
    mixture log-loss, connected by a line -- same slope-chart convention as
    final_ranking.plot_slope's raw-vs-calibrated comparison (log-loss values
    cluster near 1.0 with no meaningful zero, so a bar chart would exaggerate
    the gaps -- see CLAUDE.md)."""
    pivot = table.pivot(index="algorithm", columns="market", values="log_loss")
    algos = [a for a in ALGO_ORDER if a in pivot.index]
    n = len(algos)
    fig, ax = plt.subplots(figsize=(8, 0.6 * n + 2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ypos = np.arange(n)[::-1]
    for yp, algo in zip(ypos, algos):
        color = COLORS[algo]
        op, fu, cl = pivot.loc[algo, "opening"], pivot.loc[algo, "full"], pivot.loc[algo, "closing"]
        ax.plot([op, cl], [yp, yp], color=color, linewidth=1.6, zorder=2, alpha=0.9)
        ax.scatter([op], [yp], s=70, facecolor="#fcfcfb", edgecolor=color, linewidth=1.8, zorder=3)
        ax.scatter([fu], [yp], s=70, facecolor="#c3c2b7", edgecolor=color, linewidth=1.2, zorder=3)
        ax.scatter([cl], [yp], s=70, facecolor=color, edgecolor=color, linewidth=1.0, zorder=3)

    ax.set_yticks(ypos)
    ax.set_yticklabels(algos, fontsize=9, color="#0b0b0b")
    ax.set_xlabel("Log-loss (full history)", color="#52514e")
    ax.set_title(f"Opening vs. closing odds: mixture log-loss by algorithm ({label} de-vig)",
                  color="#0b0b0b", fontsize=12, pad=12)
    ax.grid(True, axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", left=False)

    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="#52514e", markersize=8, markeredgewidth=1.8, label="opening"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#c3c2b7",
               markeredgecolor="#52514e", markersize=8, label="full panel (26)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#52514e",
               markeredgecolor="#52514e", markersize=8, label="closing"),
    ]
    legend = ax.legend(handles=legend_elements, frameon=False, loc="best", fontsize=8.5)
    for text in legend.get_texts():
        text.set_color("#0b0b0b")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def build_combined_ranking(table):
    """Combines this file's 12 algorithm rows (4 algorithms x 3 market
    subsets) with every individual bookmaker's raw log-loss from
    sleeping_experts.py's results_table.csv into one ranked table -- the
    same "algorithms + bookmakers together, sorted by log-loss" convention
    that file already uses, just with the market-phase-split algorithm
    variants standing in for its single pooled-panel ones."""
    algo_rows = table.copy()
    algo_rows["name"] = algo_rows["algorithm"] + " (" + algo_rows["market"] + ")"
    algo_rows["type"] = "algorithm"
    algo_rows["single_season"] = False

    results_table = pd.read_csv(os.path.join(OUT_DIR, "results_table.csv"))
    book_rows = results_table[results_table["type"] == "bookmaker"].copy()
    book_rows["market"] = np.where(book_rows["name"].str.endswith("C"), "closing", "opening")

    cols = ["name", "type", "market", "rounds", "log_loss", "brier", "rps", "accuracy", "single_season"]
    combined = pd.concat([algo_rows[cols], book_rows[cols]], ignore_index=True)
    combined = combined.sort_values("log_loss").reset_index(drop=True)
    combined.insert(0, "rank", np.arange(1, len(combined) + 1))
    return combined


def plot_combined_ranking(combined, out_path):
    """Saves results/opening_vs_closing_ranking.png: one Cleveland dot per
    entry (12 algorithm variants + 26 individual bookmakers), colored by
    type (algorithm vs. bookmaker -- same RANK_COLORS convention as
    final_ranking.plot_slope) and marker-styled by market phase: hollow =
    opening, muted grey = full panel (algorithms only), filled = closing."""
    n = len(combined)
    fig, ax = plt.subplots(figsize=(9, 0.3 * n + 2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ypos = np.arange(n)[::-1]
    for yp, (_, row) in zip(ypos, combined.iterrows()):
        color = RANK_COLORS[row["type"]]
        face = {"opening": "#fcfcfb", "full": "#c3c2b7", "closing": color}[row["market"]]
        ax.scatter([row["log_loss"]], [yp], s=45, facecolor=face, edgecolor=color, linewidth=1.4, zorder=3)

    ax.set_yticks(ypos)
    ax.set_yticklabels(combined["name"], fontsize=7, color="#0b0b0b")
    ax.set_xlabel("Log-loss (full history)", color="#52514e")
    ax.set_title("Full ranking: algorithm variants (opening/closing/full panel) + every bookmaker",
                  color="#0b0b0b", fontsize=12, pad=12)
    ax.grid(True, axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", left=False)

    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="#52514e", markersize=7, markeredgewidth=1.4, label="opening"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#c3c2b7",
               markeredgecolor="#52514e", markersize=7, label="full panel (algorithms only)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#52514e",
               markeredgecolor="#52514e", markersize=7, label="closing"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=RANK_COLORS["algorithm"],
               markeredgecolor=RANK_COLORS["algorithm"], markersize=7, label="algorithm"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=RANK_COLORS["bookmaker"],
               markeredgecolor=RANK_COLORS["bookmaker"], markersize=7, label="bookmaker"),
    ]
    legend = ax.legend(handles=legend_elements, frameon=False, loc="upper right", fontsize=7.5)
    for text in legend.get_texts():
        text.set_color("#0b0b0b")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def run_opening_vs_closing(panel, P, awake, y, label="Basic", suffix="", with_ranking=True):
    """Runs the opening-only / closing-only / full-panel comparison on ONE
    de-vig'd panel. `label`/`suffix` let it run on either dataset without
    overwriting the other's outputs (Basic keeps the original, unsuffixed
    filenames; Shin writes *_shin.*).

    `with_ranking=False` skips the combined "algorithm variants + every
    individual bookmaker" ranking, which is Basic-only: it reads the
    individual-bookmaker rows out of results_table.csv, and that file is
    written by sleeping_experts_experiment.py from the BASIC dataset. Mixing
    Shin algorithm rows with Basic bookmaker rows in one ranking would be
    comparing numbers computed on two different panels."""
    print(f"\n{'#' * 70}\n1. Opening vs. closing odds -- {label} normalization\n{'#' * 70}")
    T = len(y)
    open_idx, close_idx = split_by_market_phase(panel)
    print(f"Opening-odds panel: {len(open_idx)} bookmakers -- {[panel[k] for k in open_idx]}")
    print(f"Closing-odds panel: {len(close_idx)} bookmakers -- {[panel[k] for k in close_idx]}")

    rows, ll_by_market_algo, per_round_by_metric = [], {}, {}
    for market, indices in [("opening", open_idx), ("closing", close_idx), ("full", list(range(len(panel))))]:
        P_sub, awake_sub, y_sub, keep = build_subset(P, awake, y, indices)
        phats = fit_all_algorithms(P_sub, awake_sub, y_sub)
        for algo, phat in phats.items():
            m = evaluate(phat, y_sub)
            rows.append({"market": market, "algorithm": algo, "n_bookmakers": len(indices), **m})

            idx_sub = np.arange(len(y_sub))
            ll_sub = -np.log(np.clip(phat[idx_sub, y_sub], EPS, 1.0))
            full_ll = np.full(T, np.nan)
            full_ll[keep] = ll_sub
            ll_by_market_algo[(market, algo)] = full_ll

            # same scatter for Brier and for the 0/1 "picked the right outcome"
            # indicator, so the vs-uniform test below can be run on all three
            # metrics rather than log-loss alone
            for metric, values in [
                ("log_loss", ll_sub),
                ("brier", per_round_brier(phat, y_sub, idx_sub)),
                ("accuracy", (phat.argmax(axis=1) == y_sub).astype(float)),
            ]:
                scattered = np.full(T, np.nan)
                scattered[keep] = values
                per_round_by_metric[(market, algo, metric)] = scattered

    table = pd.DataFrame(rows).sort_values(["algorithm", "market"])
    table["normalization"] = label
    table_path = os.path.join(OUT_DIR, f"opening_vs_closing_table{suffix}.csv")
    table.to_csv(table_path, index=False)
    pd.set_option("display.width", 160)
    print(f"\n=== Opening vs. closing odds ({label}), per algorithm — saved to {table_path} ===\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    sig_rows = []
    for algo in ALGO_ORDER:
        for comparison, a, b in [
            ("closing vs opening", ll_by_market_algo[("closing", algo)], ll_by_market_algo[("opening", algo)]),
            ("closing vs full", ll_by_market_algo[("closing", algo)], ll_by_market_algo[("full", algo)]),
            ("opening vs full", ll_by_market_algo[("opening", algo)], ll_by_market_algo[("full", algo)]),
        ]:
            d, n = paired_diff(a, b)
            res = moving_block_bootstrap_test(d)
            sig_rows.append({"algorithm": algo, "comparison": comparison, "n": n,
                              "normalization": label, **res})

    # Does the LEARNING actually pay off WITHIN each panel, or would naive
    # uniform averaging do just as well? On the full panel every algorithm
    # beats uniform significantly (significance_test.py) -- but that panel
    # deliberately mixes 13 closing with 13 systematically worse opening
    # bookmakers, i.e. there is a real quality spread to learn. Restricted to
    # one market phase the experts are far more homogeneous, so there may be
    # nothing left to exploit. Cheap to check here: the per-round loss arrays
    # for all 4 algorithms on all 3 panels already exist at this point.
    # Run on all three metrics, because they do NOT agree: log-loss and Brier
    # are both proper scoring rules over the whole distribution, while accuracy
    # only looks at the argmax and ignores confidence entirely.
    #
    # SIGN: mean_diff is always (algorithm - uniform). For log_loss/brier lower
    # is better, so negative = the algorithm wins; for accuracy HIGHER is
    # better, so the sign flips. Rather than make the reader track that, the
    # `verdict` column below states the winner outright (see CLAUDE.md's
    # sign-convention gotcha for why this file doesn't leave it implicit).
    LOWER_IS_BETTER = {"log_loss": True, "brier": True, "accuracy": False}

    vs_uniform_rows = []
    for market in ["opening", "closing", "full"]:
        for algo in [a for a in ALGO_ORDER if a != "Uniform average"]:
            for metric, lower_better in LOWER_IS_BETTER.items():
                d, n = paired_diff(per_round_by_metric[(market, algo, metric)],
                                    per_round_by_metric[(market, "Uniform average", metric)])
                res = moving_block_bootstrap_test(d)
                algo_wins = (res["mean_diff"] < 0) if lower_better else (res["mean_diff"] > 0)
                verdict = ("tie (not significant)" if not res["significant_95"]
                           else (f"{algo} wins" if algo_wins else "Uniform average wins"))
                vs_uniform_rows.append({
                    "market": market, "algorithm": algo, "metric": metric,
                    "comparison": f"{algo} vs Uniform average",
                    "lower_is_better": lower_better, "verdict": verdict,
                    "n": n, "normalization": label, **res,
                })

    vs_uniform = pd.DataFrame(vs_uniform_rows)
    vs_uniform_path = os.path.join(OUT_DIR, f"opening_vs_closing_vs_uniform{suffix}.csv")
    vs_uniform.to_csv(vs_uniform_path, index=False)
    print(f"\n=== Learned mixture vs. naive uniform averaging, WITHIN each panel ({label}) "
          f"— saved to {vs_uniform_path} ===")
    print("(mean_diff = algorithm - uniform; for accuracy HIGHER is better, so read `verdict`)\n")
    for metric in LOWER_IS_BETTER:
        sub = vs_uniform[vs_uniform["metric"] == metric]
        print(f"-- {metric} --")
        print(sub[["market", "algorithm", "n", "mean_diff", "ci_low", "ci_high",
                    "p_value", "verdict"]]
              .to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        print()

    sig_table = pd.DataFrame(sig_rows)
    sig_path = os.path.join(OUT_DIR, f"opening_vs_closing_significance{suffix}.csv")
    sig_table.to_csv(sig_path, index=False)
    print(f"\n=== Significance, {label} (negative mean_diff = first-named is better) "
          f"— saved to {sig_path} ===\n")
    print(sig_table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    plot_market_comparison(table, os.path.join(OUT_DIR, f"opening_vs_closing{suffix}.png"), label)

    if not with_ranking:
        return table, sig_table

    combined = build_combined_ranking(table)
    ranking_path = os.path.join(OUT_DIR, "opening_vs_closing_ranking.csv")
    combined.to_csv(ranking_path, index=False)
    print(f"\n=== Combined ranking: algorithm variants + individual bookmakers — saved to {ranking_path} ===\n")
    print(combined.drop(columns="single_season").to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    single_season = combined[combined["single_season"]]
    if not single_season.empty:
        print(f"\n=== WARNING: {len(single_season)} bookmakers below are active in only ONE season -- "
              f"their rank is confounded with that season's difficulty, not a trustworthy skill "
              f"comparison against the rest (see results_table.csv) ===\n")
        print(single_season[["rank", "name", "log_loss"]].to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    plot_combined_ranking(combined, os.path.join(OUT_DIR, "opening_vs_closing_ranking.png"))


# =====================================================================
# 2. opening/closing quality transfer correlation
# =====================================================================

def build_pair_table(panel):
    """One row per opening/closing bookmaker pair (13 total): both sides'
    raw log-loss / coverage / single_season flag, read from
    results_table.csv (already computed the standard causal way -- no new
    algorithm run needed for this descriptive check)."""
    results = pd.read_csv(os.path.join(OUT_DIR, "results_table.csv")).set_index("name")
    open_idx, close_idx = split_by_market_phase(panel)
    close_names = [panel[k] for k in close_idx]

    rows = []
    for close_name in close_names:
        open_name = close_name[:-1]
        if open_name not in results.index:
            continue  # unpaired closing series, already warned about by split_by_market_phase
        o, c = results.loc[open_name], results.loc[close_name]
        rows.append({
            "pair": open_name, "opening_name": open_name, "closing_name": close_name,
            "opening_log_loss": o["log_loss"], "closing_log_loss": c["log_loss"],
            "opening_coverage_pct": o["coverage_pct"], "closing_coverage_pct": c["coverage_pct"],
            "either_single_season": bool(o["single_season"] or c["single_season"]),
        })
    return pd.DataFrame(rows)


def correlate(x, y):
    """Pearson (raw values) and Spearman (rank) correlation, via plain numpy
    -- no scipy dependency needed (matches this project's convention of not
    requiring anything beyond pandas/numpy/matplotlib)."""
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])
    return pearson, spearman


def plot_quality_transfer(pairs, out_path):
    """Scatter: each bookmaker pair's closing-side log-loss (x) vs.
    opening-side log-loss (y), + a least-squares trend line. A positive
    slope / high correlation would mean "sharp at closing predicts sharp at
    opening" -- direct visual answer to the motivating question."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    multi = pairs[~pairs["either_single_season"]]
    single = pairs[pairs["either_single_season"]]
    ax.scatter(multi["closing_log_loss"], multi["opening_log_loss"], s=70,
                color="#2a78d6", edgecolor="#fcfcfb", linewidth=1.0, zorder=3, label="multi-season pair")
    ax.scatter(single["closing_log_loss"], single["opening_log_loss"], s=70,
                color="#c3c2b7", edgecolor="#fcfcfb", linewidth=1.0, zorder=3, label="single-season pair (either side)")

    for _, row in pairs.iterrows():
        ax.annotate(row["pair"], (row["closing_log_loss"], row["opening_log_loss"]),
                     xytext=(5, 4), textcoords="offset points", fontsize=7.5, color="#52514e")

    x = pairs["closing_log_loss"].values
    y = pairs["opening_log_loss"].values
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, slope * xs + intercept, color="#898781", linewidth=1.5, linestyle="--", zorder=2,
             label="least-squares trend")

    ax.set_xlabel("Closing-side log-loss (full history)", color="#52514e")
    ax.set_ylabel("Opening-side log-loss (full history)", color="#52514e")
    ax.set_title("Does closing-side sharpness predict opening-side sharpness?\n(one point per bookmaker pair)",
                  color="#0b0b0b", fontsize=12, pad=12)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")
    legend = ax.legend(frameon=False, loc="best", fontsize=8.5)
    for text in legend.get_texts():
        text.set_color("#0b0b0b")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def run_quality_transfer_correlation(panel):
    print(f"\n{'#' * 70}\n2. Opening/closing quality-transfer correlation\n{'#' * 70}")
    pairs = build_pair_table(panel)
    pearson_all, spearman_all = correlate(pairs["closing_log_loss"], pairs["opening_log_loss"])
    multi = pairs[~pairs["either_single_season"]]
    pearson_multi, spearman_multi = correlate(multi["closing_log_loss"], multi["opening_log_loss"])

    pairs_path = os.path.join(OUT_DIR, "opening_closing_quality_transfer.csv")
    pairs.to_csv(pairs_path, index=False)
    pd.set_option("display.width", 160)
    print(f"=== Opening vs. closing log-loss, per bookmaker pair (n={len(pairs)}) -- saved to {pairs_path} ===\n")
    print(pairs.sort_values("closing_log_loss").to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nAll {len(pairs)} pairs:          Pearson r={pearson_all:+.3f}, Spearman rho={spearman_all:+.3f}")
    print(f"Multi-season pairs only (n={len(multi)}): Pearson r={pearson_multi:+.3f}, Spearman rho={spearman_multi:+.3f}")

    plot_quality_transfer(pairs, os.path.join(OUT_DIR, "opening_closing_quality_transfer.png"))


# =====================================================================
# 3. shin vs. basic: log-loss / Brier comparison
# =====================================================================

def build_entries(panel, P, awake, y):
    """Mirrors final_ranking.py's entries construction: the 4 algorithm
    mixtures (always awake) plus every Tier-A bookmaker (coverage_pct >=
    COVERAGE_THRESHOLD), each as {name: (probs, mask, kind)}."""
    T = P.shape[0]
    entries = {}
    for name, phat in fit_all_algorithms(P, awake, y).items():
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
    version -- one calibrate_full_history() call serves both metrics."""
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
    final_ranking.plot_slope(), colored by type with RANK_COLORS."""
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


def run_shin_vs_basic_comparison():
    print(f"\n{'#' * 70}\n3. Shin vs. Basic: log-loss / Brier comparison\n{'#' * 70}")
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
    # the favorite-longshot bias, or does it still add something? ----
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


# =====================================================================
# 4. shin vs. basic: calibration quality (ECE / reliability)
# =====================================================================

def calibration_for_dataset(panel, P, awake, y):
    """Raw (uncalibrated) calibration tables for the 4 algorithms +
    REFERENCE_BOOKMAKERS on one panel -- same entity set as
    calibration_analysis.main()."""
    tables, ece_rows = {}, []
    for name, phat in fit_all_algorithms(P, awake, y).items():
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
    calibration_analysis.plot_reliability), one for Basic normalization and
    one for Shin, sharing a y-axis so the flattening (or not) is directly
    comparable at a glance. Same series colors as calibration_analysis.py."""
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


def run_shin_vs_basic_calibration():
    print(f"\n{'#' * 70}\n4. Shin vs. Basic: calibration quality (ECE / reliability)\n{'#' * 70}")
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


# =====================================================================

def main():
    panel, P, awake, y, dates, seasons = load_full_universe()

    run_opening_vs_closing(panel, P, awake, y)

    # the same market-phase split, rerun on the Shin-de-vigged panel: does the
    # "closing < full < opening" ordering survive a different de-vig formula, or
    # is it an artifact of proportional normalization? (Experiment 3 asks the
    # same question of the pooled panel; this asks it of the phase split.)
    panel_s, P_s, awake_s, y_s, _, _ = load_universe_from(
        os.path.join(DATA_DIR, "odds_long_shin.csv"))
    run_opening_vs_closing(panel_s, P_s, awake_s, y_s,
                            label="Shin", suffix="_shin", with_ranking=False)

    run_quality_transfer_correlation(panel)
    run_shin_vs_basic_comparison()
    run_shin_vs_basic_calibration()


if __name__ == "__main__":
    main()
