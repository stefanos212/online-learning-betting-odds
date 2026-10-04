"""
Is there a genuine (non-leaky) Kelly betting edge against bookmakers OTHER
than the two value_betting.py originally tested (B365C, BW)? Extends the
same leave-one-bookmaker-out design to every sufficiently-covered,
multi-season bookmaker (coverage_pct >= MIN_COVERAGE_PCT, not single-season
-- see results_table.csv), using the methodology appropriate to each
bookmaker's own market phase:

  - CLOSING-side held-out bookmakers: the standard leave-one-out panel
    (exclude only that column from the full 26-bookmaker panel) -- same as
    value_betting.py's own B365C test. Valid because every OTHER
    bookmaker's closing price genuinely settles at or before roughly the
    same time (near kickoff) as this one's, so using them as mixture inputs
    isn't a look-ahead artifact.
  - OPENING-side held-out bookmakers: the CORRECTED opening-only leave-one-
    out panel (see opening_closing_quality_transfer.py's module docstring
    for the full argument) -- excludes every closing-suffixed bookmaker
    too, since none of that information would exist yet when only an
    opening price is posted. value_betting.py's original BW test used the
    leaky (closing-inclusive) version of this and found a "huge" but,
    per opening_closing_quality_transfer.py, spurious edge.

Only OGD is tested (matching value_betting.py's own scope -- the only one
of the 4 algorithms significance_test.py found to significantly beat
uniform averaging).

Produces:
  - results/value_betting_all_bookmakers_table.csv
  - results/value_betting_all_bookmakers.png  (one row per bookmaker: open
    circle = raw mean log-growth/bet, filled = calibrated; vertical dashed
    line at 0 = breakeven)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import OUT_DIR, load_full_universe, per_round_expert_loss, run_ogd_sleeping
from final_ranking import calibrate_full_history
from significance_test import moving_block_bootstrap_test
from opening_vs_closing import split_by_market_phase
from value_betting import (
    load_match_order, load_decimal_odds, panel_excluding, simulate_kelly_betting, STARTING_BANKROLL,
)

MIN_COVERAGE_PCT = 50.0


def select_bookmakers():
    """Every bookmaker with coverage_pct >= MIN_COVERAGE_PCT that isn't
    flagged single_season (see results_table.csv / CLAUDE.md gotcha) --
    thin/single-season entries would give too few bets for a meaningful
    bootstrap test and their apparent performance is confounded with one
    season's difficulty anyway."""
    results = pd.read_csv(os.path.join(OUT_DIR, "results_table.csv"))
    books = results[(results["type"] == "bookmaker") &
                     (results["coverage_pct"] >= MIN_COVERAGE_PCT) &
                     (~results["single_season"])]
    return books.sort_values("coverage_pct", ascending=False)["name"].tolist()


def simulate_and_score(phat_raw, phat_cal, odds, y, book_mask, book, method):
    """Runs the Kelly simulation for raw + calibrated p_hat against one
    bookmaker's real odds, returns the two result rows -- shared scoring
    logic for both the closing and opening test paths below."""
    rows = []
    for stage, p_hat in [("raw", phat_raw), ("calibrated", phat_cal)]:
        growth, bet_placed = simulate_kelly_betting(p_hat, odds, y, book_mask)
        bankroll = STARTING_BANKROLL * np.cumprod(growth[book_mask])
        log_growth_per_bet = np.log(growth[bet_placed])
        n_bets = int(bet_placed.sum())
        n_active = int(book_mask.sum())
        if n_bets >= 30:
            res = moving_block_bootstrap_test(log_growth_per_bet)
        else:
            res = {"mean_diff": log_growth_per_bet.mean() if n_bets else np.nan,
                   "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "significant_95": False}
        rows.append({
            "bookmaker": book, "method": method, "stage": stage,
            "active_rounds": n_active, "n_bets": n_bets, "bet_rate_pct": 100 * n_bets / max(1, n_active),
            "final_bankroll": bankroll[-1] if len(bankroll) else np.nan,
            "mean_log_growth_per_bet": res["mean_diff"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
            "p_value": res["p_value"], "significant_95": res["significant_95"],
        })
    return rows


def test_closing_bookmaker(panel, P, awake, y, book, match_ids):
    """Standard leave-one-out: drop only `book` from the full 26-bookmaker
    panel (value_betting.panel_excluding), fit OGD, bet against its real
    closing odds. No look-ahead concern for a closing-side target (see
    module docstring)."""
    sub_panel, P_sub, awake_sub = panel_excluding(panel, P, awake, book)
    N_sub = len(sub_panel)
    loss_sub = per_round_expert_loss(P_sub, y)
    W_ogd = run_ogd_sleeping(loss_sub, awake_sub, N_sub)
    phat_raw = np.einsum("tn,tnk->tk", W_ogd, P_sub)

    all_true = np.ones(len(y), dtype=bool)
    _, phat_cal, _, _ = calibrate_full_history(phat_raw, y, all_true)

    odds = load_decimal_odds(book, match_ids)
    book_mask = awake[:, panel.index(book)] & ~np.isnan(odds[:, 0])
    return simulate_and_score(phat_raw, phat_cal, odds, y, book_mask, book, "closing (standard leave-one-out)")


def test_opening_bookmaker(panel, P, awake, y, book, open_idx, match_ids):
    """Corrected leave-one-out: fit OGD on ONLY the other opening-side
    bookmakers (excludes every closing-suffixed one too), bet against
    `book`'s real opening odds -- see
    opening_closing_quality_transfer.run_opening_panel_bw_test, same logic
    generalized to any opening bookmaker."""
    book_col = panel.index(book)
    keep_cols = [k for k in open_idx if k != book_col]
    sub_panel = [panel[k] for k in keep_cols]
    N_sub = len(sub_panel)

    P_sub_full, awake_sub_full = P[:, keep_cols, :], awake[:, keep_cols]
    odds_full = load_decimal_odds(book, match_ids)
    book_mask_full = awake[:, book_col] & ~np.isnan(odds_full[:, 0])

    keep = awake_sub_full.any(axis=1) & book_mask_full
    P_sub, awake_sub, y_sub = P_sub_full[keep], awake_sub_full[keep], y[keep]
    odds, book_mask = odds_full[keep], np.ones(keep.sum(), dtype=bool)

    loss_sub = per_round_expert_loss(P_sub, y_sub)
    W_ogd = run_ogd_sleeping(loss_sub, awake_sub, N_sub)
    phat_raw = np.einsum("tn,tnk->tk", W_ogd, P_sub)

    all_true = np.ones(len(y_sub), dtype=bool)
    _, phat_cal, _, _ = calibrate_full_history(phat_raw, y_sub, all_true)
    return simulate_and_score(phat_raw, phat_cal, odds, y_sub, book_mask, book, "opening (corrected leave-one-out)")


def plot_edge_sweep(table, out_path):
    """Saves value_betting_all_bookmakers.png: one row per bookmaker
    (sorted by calibrated mean log-growth/bet), open circle = raw, filled =
    calibrated, vertical dashed line at 0 = breakeven (no edge). Bold label
    + asterisk marks bookmakers where the calibrated result is significant
    at 95%. Not a log-loss ranking, so no dot/slope-vs-bar concern here --
    this is genuinely a before/after (raw vs. calibrated) comparison in the
    same spirit as final_ranking.plot_slope."""
    raw = table[table["stage"] == "raw"].set_index("bookmaker")
    cal = table[table["stage"] == "calibrated"].set_index("bookmaker")
    order = cal.sort_values("mean_log_growth_per_bet").index.tolist()

    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(order) + 2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axvline(0, color="#898781", linewidth=1.3, linestyle="--", zorder=1)

    ypos = np.arange(len(order))
    for yp, book in zip(ypos, order):
        r, c = raw.loc[book, "mean_log_growth_per_bet"], cal.loc[book, "mean_log_growth_per_bet"]
        method = raw.loc[book, "method"]
        color = "#eb6834" if "opening" in method else "#2a78d6"
        ax.plot([r, c], [yp, yp], color=color, linewidth=1.5, zorder=2, alpha=0.85)
        ax.scatter([r], [yp], s=70, facecolor="#fcfcfb", edgecolor=color, linewidth=1.8, zorder=3)
        ax.scatter([c], [yp], s=70, facecolor=color, edgecolor=color, linewidth=1.0, zorder=3)

    labels = []
    for book in order:
        sig = cal.loc[book, "significant_95"]
        phase = "opening" if "opening" in raw.loc[book, "method"] else "closing"
        labels.append(f"{book} ({phase}){' *' if sig else ''}")

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.5, color="#0b0b0b")
    ax.set_xlabel("Mean log-growth per bet (0 = breakeven, quarter-Kelly)", color="#52514e")
    ax.set_title("Value-betting edge sweep: raw vs. calibrated, honest leave-one-out per bookmaker\n"
                  "(* = calibrated result significant at 95%)", color="#0b0b0b", fontsize=11.5, pad=12)
    ax.grid(True, axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", left=False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="#52514e", markersize=8, markeredgewidth=1.8, label="raw"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#52514e",
               markeredgecolor="#52514e", markersize=8, label="calibrated"),
        Line2D([0], [0], color="#eb6834", linewidth=2.5, label="opening (corrected leave-one-out)"),
        Line2D([0], [0], color="#2a78d6", linewidth=2.5, label="closing (standard leave-one-out)"),
    ]
    legend = ax.legend(handles=legend_elements, frameon=False, loc="best", fontsize=8)
    for text in legend.get_texts():
        text.set_color("#0b0b0b")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def main():
    panel, P, awake, y, dates, seasons = load_full_universe()
    open_idx, close_idx = split_by_market_phase(panel)
    match_ids = load_match_order()
    assert len(match_ids) == P.shape[0], "match_ids ordering must line up with load_full_universe()"

    books = select_bookmakers()
    print(f"Testing {len(books)} bookmakers (coverage >= {MIN_COVERAGE_PCT}%, multi-season): {books}\n")

    all_rows = []
    for book in books:
        is_closing = book.endswith("C")
        print(f"--- {book} ({'closing' if is_closing else 'opening'}) ---")
        if is_closing:
            rows = test_closing_bookmaker(panel, P, awake, y, book, match_ids)
        else:
            rows = test_opening_bookmaker(panel, P, awake, y, book, open_idx, match_ids)
        all_rows.extend(rows)
        for r in rows:
            flag = "SIGNIFICANT" if r["significant_95"] else "not significant"
            print(f"  [{r['stage']:10s}] bets={r['n_bets']:5d} ({r['bet_rate_pct']:.1f}% of "
                  f"{r['active_rounds']} active rounds)  final_bankroll={r['final_bankroll']:.4f}  "
                  f"mean_log_growth/bet={r['mean_log_growth_per_bet']:+.5f}  "
                  f"p={r['p_value']}  {flag}")
        print()

    table = pd.DataFrame(all_rows)
    out_path = os.path.join(OUT_DIR, "value_betting_all_bookmakers_table.csv")
    table.to_csv(out_path, index=False)

    pd.set_option("display.width", 160)
    print(f"=== Full sweep, {len(books)} bookmakers -- saved to {out_path} ===\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    sig = table[(table["stage"] == "calibrated") & (table["significant_95"])]
    print(f"\n=== Bookmakers with a SIGNIFICANT calibrated edge: {len(sig)} of {len(books)} ===")
    if not sig.empty:
        print(sig[["bookmaker", "method", "mean_log_growth_per_bet", "p_value"]]
              .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    plot_edge_sweep(table, os.path.join(OUT_DIR, "value_betting_all_bookmakers.png"))


if __name__ == "__main__":
    main()
