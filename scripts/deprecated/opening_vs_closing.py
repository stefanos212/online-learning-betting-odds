"""
OPENING vs. CLOSING odds: does the market's final, most-informed price make
a better mixture than its first, least-informed one?

football-data.co.uk quotes two prices per bookmaker for most of the panel:
an OPENING price (column prefix "{book}", e.g. B365H/D/A) taken when the
market first opens, and a CLOSING price (column prefix "{book}C", e.g.
B365CH/CD/CA) taken just before kickoff, after all the money has moved the
line. process_data.py's auto-detection (find_odds) preserves this
"C" suffix verbatim into the processed bookmaker name -- including through
the VC/BV rebrand merge (VC_BV = opening, VC_BVC = closing) -- so the
current 26-bookmaker panel already splits cleanly in half: 13 opening names,
13 closing counterparts, one pair per underlying bookmaker.

This file partitions load_full_universe()'s panel by that naming rule and
reruns each of the 4 mixture algorithms (Hedge / OGD / FTRL / Uniform
average) on THREE panels: restricted to only the opening-odds bookmakers,
restricted to only the closing-odds ones, and the FULL 26-bookmaker panel
(opening+closing mixed -- the same one results_table.csv already reports,
recomputed here so it can be paired on a shared round index against the two
subsets). Same sleeping-experts machinery (run_hedge_sleeping /
run_ogd_sleeping / run_ftrl_sleeping / evaluate) reused unchanged -- these
functions already accept an arbitrary N-sized panel and don't care what the
columns represent, so no new algorithm code is needed, only a restricted
view of the panel plus a re-applied "at least one expert in this subset was
awake" reduction (same trick load_full_universe() itself uses at the
full-panel level; for the full panel this reduction is a no-op, since
load_full_universe() already guarantees every round has >=1 of 26 awake).

Motivating question for the third (full) panel: is closing-only actually
BETTER than throwing every bookmaker (opening included) into one mixture,
or does it just look better than opening alone while still losing to the
full panel? Point estimates suggested closing-only beats the full panel for
all 4 algorithms -- this file adds the bootstrap tests to confirm that
isn't sampling noise.

Produces:
  - results/opening_vs_closing_table.csv          (per algorithm x market
                                                      {opening, closing,
                                                      full}: rounds,
                                                      log-loss, Brier, RPS,
                                                      accuracy)
  - results/opening_vs_closing_significance.csv   (bootstrap tests, per
                                                      algorithm: closing vs
                                                      opening, closing vs
                                                      full, opening vs full)
  - results/opening_vs_closing.png                (slope chart, one row per
                                                      algorithm: open circle
                                                      = opening, muted-grey
                                                      = full panel, filled
                                                      = closing)
  - results/opening_vs_closing_ranking.csv/.png   (the 12 algorithm-variant
                                                      rows above combined
                                                      with every individual
                                                      bookmaker's raw
                                                      log-loss from
                                                      sleeping_experts.py's
                                                      results_table.csv, one
                                                      ranked table/dot plot)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sleeping_experts import (
    OUT_DIR, EPS, COLORS,
    load_full_universe, per_round_expert_loss,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping, evaluate,
)
from significance_test import moving_block_bootstrap_test, paired_diff
from final_ranking import RANK_COLORS

ALGO_ORDER = ["Hedge", "OGD", "FTRL", "Uniform average"]


def split_by_market_phase(panel):
    """A processed bookmaker name is the CLOSING-odds counterpart of the
    identically-prefixed OPENING one iff it ends in "C" -- guaranteed by
    football-data.co.uk's own column convention, which process_data.py's
    auto-detected prefixes preserve verbatim (see module docstring). Returns
    (open_idx, close_idx), the panel indices for each subset."""
    close_idx = [k for k, name in enumerate(panel) if name.endswith("C")]
    open_idx = [k for k, name in enumerate(panel) if not name.endswith("C")]
    unpaired = [panel[k] for k in close_idx if panel[k][:-1] not in panel]
    if unpaired:
        print(f"  NOTE: closing-odds series with no matching opening name: {unpaired}")
    return open_idx, close_idx


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


def run_all_algorithms(P_sub, awake_sub, y_sub):
    """Runs the 4 mixture algorithms on one subset's reduced panel, returns
    {algorithm_name: phat (T_sub, 3)}."""
    N_sub = P_sub.shape[1]
    loss = per_round_expert_loss(P_sub, y_sub)
    algo_weights = {
        "Hedge": run_hedge_sleeping(loss, awake_sub, N_sub),
        "OGD": run_ogd_sleeping(loss, awake_sub, N_sub),
        "FTRL": run_ftrl_sleeping(loss, awake_sub, N_sub),
        "Uniform average": awake_sub / awake_sub.sum(axis=1, keepdims=True),
    }
    return {name: np.einsum("tn,tnk->tk", W, P_sub) for name, W in algo_weights.items()}


def plot_market_comparison(table, out_path):
    """Saves results/opening_vs_closing.png: one row per algorithm, open
    circle = opening-odds mixture log-loss, filled circle = closing-odds
    mixture log-loss, connected by a line -- same slope-chart convention as
    final_ranking.plot_slope's raw-vs-calibrated comparison (log-loss values
    cluster near 1.0 with no meaningful zero, so a bar chart would exaggerate
    the gaps -- see CLAUDE.md), reused here for opening-vs-closing."""
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
    ax.set_title("Opening vs. closing odds: mixture log-loss by algorithm",
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


def main():
    panel, P, awake, y, _, _ = load_full_universe()
    T = len(y)
    open_idx, close_idx = split_by_market_phase(panel)
    print(f"Opening-odds panel: {len(open_idx)} bookmakers -- {[panel[k] for k in open_idx]}")
    print(f"Closing-odds panel: {len(close_idx)} bookmakers -- {[panel[k] for k in close_idx]}")

    rows, ll_by_market_algo = [], {}
    for market, indices in [("opening", open_idx), ("closing", close_idx), ("full", list(range(len(panel))))]:
        P_sub, awake_sub, y_sub, keep = build_subset(P, awake, y, indices)
        phats = run_all_algorithms(P_sub, awake_sub, y_sub)
        for algo, phat in phats.items():
            m = evaluate(phat, y_sub)
            rows.append({"market": market, "algorithm": algo, "n_bookmakers": len(indices), **m})

            ll_sub = -np.log(np.clip(phat[np.arange(len(y_sub)), y_sub], EPS, 1.0))
            full_ll = np.full(T, np.nan)
            full_ll[keep] = ll_sub
            ll_by_market_algo[(market, algo)] = full_ll

    table = pd.DataFrame(rows).sort_values(["algorithm", "market"])
    table_path = os.path.join(OUT_DIR, "opening_vs_closing_table.csv")
    table.to_csv(table_path, index=False)
    pd.set_option("display.width", 160)
    print(f"\n=== Opening vs. closing odds, per algorithm — saved to {table_path} ===\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    sig_rows = []
    for algo in ALGO_ORDER:
        for label, a, b in [
            ("closing vs opening", ll_by_market_algo[("closing", algo)], ll_by_market_algo[("opening", algo)]),
            ("closing vs full", ll_by_market_algo[("closing", algo)], ll_by_market_algo[("full", algo)]),
            ("opening vs full", ll_by_market_algo[("opening", algo)], ll_by_market_algo[("full", algo)]),
        ]:
            d, n = paired_diff(a, b)
            res = moving_block_bootstrap_test(d)
            sig_rows.append({"algorithm": algo, "comparison": label, "n": n, **res})

    sig_table = pd.DataFrame(sig_rows)
    sig_path = os.path.join(OUT_DIR, "opening_vs_closing_significance.csv")
    sig_table.to_csv(sig_path, index=False)
    print(f"\n=== Significance (negative mean_diff = first-named is better) — saved to {sig_path} ===\n")
    print(sig_table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    plot_market_comparison(table, os.path.join(OUT_DIR, "opening_vs_closing.png"))

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


if __name__ == "__main__":
    main()
