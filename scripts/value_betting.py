"""
VALUE-BETTING / ECONOMIC ANALYSIS -- every value-betting experiment in the
project, consolidated into one file (only value_betting_online_kelly.py
stays separate, since it swaps out the STAKING mechanism itself rather than
varying the training panel or normalization).

Question this file answers, in various framings: "if our mixture's
probabilities are a little better than a bookmaker's, would actually
betting against that bookmaker's odds make money, once the bookmaker's own
margin (vig) is accounted for?" This is DIFFERENT from "is our log-loss/
Brier/RPS better" -- a proper-scoring-rule improvement does not
automatically imply an exploitable betting edge, because the vig has to be
overcome too, not just beaten in a paper metric.

METHODOLOGY -- avoiding two kinds of leakage:
  - CIRCULARITY: if the held-out bookmaker's own price is one of the
    experts INSIDE the mixture, "betting against it" partly beats it with
    its own information. Fixed by a leave-one-bookmaker-out design: exclude
    the target from the training panel before fitting.
  - LOOK-AHEAD BIAS: excluding just the target's own column isn't enough
    when the target is an OPENING-odds bookmaker -- the rest of a 26-
    bookmaker panel still includes every CLOSING-odds bookmaker, and none
    of that closing-side information exists yet when only an opening price
    is posted (all bookmakers' closing prices settle near kickoff,
    regardless of when each one opened). So an opening-side target needs
    the training panel restricted to OTHER OPENING bookmakers only, not
    just "the full panel minus this one column". A closing-side target has
    no such problem (every closing price settles at roughly the same time).

The six experiments below (`run_*`), in the order `main()` runs them:
  1. `run_original_two`       -- the original 2-bookmaker test (B365C sharp,
                                  BW weak), full-panel-minus-target. BW's
                                  result here is the KNOWN-LEAKY one, kept
                                  for comparison against #4's correction.
  2. `run_all_bookmakers_sweep` -- extends #1's design (properly, per target
                                  phase) to every sufficiently-covered
                                  bookmaker (11 total).
  3. `run_closing_trained_both_norms` -- closing-only training panel vs.
                                  closing-only targets, Basic + Shin.
  4. `run_bw_leakage_check`   -- quantifies #1's BW leak: honest
                                  (opening-only, BW excluded) vs. circular
                                  (opening-only, BW INCLUDED -- a sanity
                                  check, not a valid estimate) vs. the
                                  original leaky number.
  5. `run_training_panel_sweep` -- the full grid: 2 normalizations x 3
                                  training-panel types x every applicable
                                  target = 108 rows.
  6. `run_shin_vs_basic`      -- does Shin normalization change the
                                  economics for the two WORST algorithms
                                  (FTRL, Uniform average)? Only place in
                                  this file that uses an algorithm other
                                  than OGD.

Only OGD is used in experiments 1-5 (significance_test.py found it's the
only one of the four mixture algorithms that significantly beats uniform
averaging, so it's the only "candidate" worth putting money behind).

Produces (one CSV/PNG pair per experiment, see each run_* docstring):
  results/value_betting_table.csv, .png
  results/value_betting_all_bookmakers_table.csv, .png
  results/value_betting_closing_trained_table.csv, .png
  results/opening_closing_quality_transfer.csv, .png (correlation half lives
    in market_devig_comparisons.py; the value-betting half is here)
  results/opening_value_betting_corrected.csv, .png
  results/value_betting_training_panel_sweep.csv
  results/shin_vs_basic_value_betting_table.csv, .png,
    shin_vs_basic_value_betting_significance.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sleeping_experts import (
    DATA_DIR, OUT_DIR, EPS, COLORS, OUTCOME_INDEX, EXCLUDED_BOOKMAKERS,
    per_round_expert_loss, run_ogd_sleeping, run_ftrl_sleeping,
    load_match_order, load_universe_from, split_by_market_phase,
)
from betting import (
    KELLY_MULT, STARTING_BANKROLL, MIN_COVERAGE_PCT,
    load_decimal_odds, select_bookmakers, kelly_fraction,
    simulate_kelly_betting, fit_leave_out, score_bet, _style_axes,
)
from final_ranking import calibrate_full_history
from significance_test import moving_block_bootstrap_test

HELD_OUT_BOOKMAKERS = ["B365C", "BW"]  # B365C = sharp/toughest test, BW = weaker/easier contrast
HELD_OUT_OPENING_BOOKMAKER = "BW"      # the one opening-side test above, examined in depth in #4
WORST_ALGORITHMS = ["FTRL", "Uniform average"]  # for #6 -- the 2 worst of the 4 by calibrated log-loss


# =====================================================================

# =====================================================================
# 1. original two-bookmaker test (B365C, BW) -- BW's result here is the
#    KNOWN-LEAKY one; see run_bw_leakage_check for the corrected version.
# =====================================================================

def run_original_two(panel, P, awake, y, match_ids):
    """Full-panel-minus-target leave-one-out for B365C and BW. Produces
    results/value_betting_table.csv, value_betting.png."""
    print(f"\n{'#' * 70}\n1. Original two-bookmaker test (B365C, BW)\n{'#' * 70}")
    N = len(panel)
    rows, curves = [], {}

    for book in HELD_OUT_BOOKMAKERS:
        print(f"\n--- Held out: {book} ---")
        _, phat_raw, phat_cal, odds, book_mask, y_sub, best_c = fit_leave_out(
            panel, P, awake, y, list(range(N)), book, match_ids, apply_reduction=False
        )
        print(f"  calibration learning-rate scale chosen on train: {best_c}")

        for stage, p_hat in [("raw", phat_raw), ("calibrated", phat_cal)]:
            growth, bet_placed = simulate_kelly_betting(p_hat, odds, y_sub, book_mask)
            row, bankroll = score_bet(growth, bet_placed, book_mask,
                                       {"held_out_bookmaker": book, "stage": stage})
            rows.append(row)
            curves[f"{book} ({stage})"] = bankroll
            print(f"  [{stage:10s}] bets={row['n_bets']:5d} ({row['bet_rate_pct']:.1f}% of "
                  f"{row['active_rounds']} active rounds)  final_bankroll={row['final_bankroll']:.3f}  "
                  f"mean_log_growth/bet={row['mean_log_growth_per_bet']:+.5f}  p={row['p_value']}")

    table = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "value_betting_table.csv")
    table.to_csv(out_path, index=False)
    pd.set_option("display.width", 160)
    print(f"\n=== Value-betting results — saved to {out_path} ===\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    plot_bankrolls_simple(curves, "Quarter-Kelly bankroll simulation, leave-one-bookmaker-out",
                           os.path.join(OUT_DIR, "value_betting.png"))
    return table


def plot_bankrolls_simple(curves, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axhline(1.0, color="#898781", linewidth=1.2, linestyle="--", zorder=1)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
    for (label, bankroll), color in zip(curves.items(), palette):
        ax.plot(np.arange(1, len(bankroll) + 1), bankroll, label=label, color=color, linewidth=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("Bet number (chronological, only rounds with a placed bet)", color="#52514e")
    ax.set_ylabel("Bankroll (log scale, start = 1.0)", color="#52514e")
    ax.set_title(title, color="#0b0b0b", fontsize=12, pad=12)
    _style_axes(ax)
    legend = ax.legend(frameon=False, loc="best", fontsize=8.5)
    for text in legend.get_texts():
        text.set_color("#0b0b0b")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


# =====================================================================
# 2. full sweep across every sufficiently-covered bookmaker, methodology
#    matched to each target's own market phase.
# =====================================================================

def run_all_bookmakers_sweep(panel, P, awake, y, open_idx, close_idx, match_ids):
    """Closing-side targets: standard leave-one-out (full panel minus
    target -- valid, every closing price settles near kickoff regardless of
    when it opened). Opening-side targets: corrected leave-one-out (opening
    bookmakers only, target excluded -- excludes every closing-suffixed
    bookmaker too, since none of that would exist yet at opening-bet time).
    Produces results/value_betting_all_bookmakers_table.csv, .png."""
    print(f"\n{'#' * 70}\n2. Full sweep across sufficiently-covered bookmakers\n{'#' * 70}")
    N = len(panel)
    books = select_bookmakers()
    print(f"Testing {len(books)} bookmakers (coverage >= {MIN_COVERAGE_PCT}%, multi-season): {books}\n")

    rows = []
    for book in books:
        is_closing = book.endswith("C")
        indices = list(range(N)) if is_closing else open_idx
        method = "closing (standard leave-one-out)" if is_closing else "opening (corrected leave-one-out)"
        print(f"--- {book} ({'closing' if is_closing else 'opening'}) ---")

        _, phat_raw, phat_cal, odds, book_mask, y_sub, _ = fit_leave_out(
            panel, P, awake, y, indices, book, match_ids, apply_reduction=not is_closing
        )
        for stage, p_hat in [("raw", phat_raw), ("calibrated", phat_cal)]:
            growth, bet_placed = simulate_kelly_betting(p_hat, odds, y_sub, book_mask)
            row, _ = score_bet(growth, bet_placed, book_mask,
                                {"bookmaker": book, "method": method, "stage": stage})
            rows.append(row)
            flag = "SIGNIFICANT" if row["significant_95"] else "not significant"
            print(f"  [{stage:10s}] bets={row['n_bets']:5d} ({row['bet_rate_pct']:.1f}% of "
                  f"{row['active_rounds']} active rounds)  final_bankroll={row['final_bankroll']:.4f}  "
                  f"mean_log_growth/bet={row['mean_log_growth_per_bet']:+.5f}  p={row['p_value']}  {flag}")
        print()

    table = pd.DataFrame(rows)
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
    return table


def plot_edge_sweep(table, out_path):
    """Saves value_betting_all_bookmakers.png: one row per bookmaker
    (sorted by calibrated mean log-growth/bet), open circle = raw, filled =
    calibrated, vertical dashed line at 0 = breakeven (no edge). Asterisk
    marks bookmakers where the calibrated result is significant at 95%."""
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


# =====================================================================
# 3. closing-on-closing, standalone, both normalizations
# =====================================================================

def run_closing_trained_both_norms(match_ids):
    """OGD trained on ONLY the other closing-side bookmakers, bet against
    every closing-side target, for both Basic and Shin normalization.
    Causally valid, no look-ahead risk (see module docstring). Produces
    results/value_betting_closing_trained_table.csv, .png."""
    print(f"\n{'#' * 70}\n3. Closing-on-closing, Basic + Shin\n{'#' * 70}")
    books = select_bookmakers()
    targets = [b for b in books if b.endswith("C")]
    print(f"Closing-side targets: {targets}\n")

    all_rows = []
    curves_by_target = {t: {} for t in targets}

    for norm_label, path in [("Basic", os.path.join(DATA_DIR, "odds_long.csv")),
                              ("Shin", os.path.join(DATA_DIR, "odds_long_shin.csv"))]:
        print(f"\n{'=' * 60}\n{norm_label} normalization\n{'=' * 60}")
        panel, P, awake, y, dates, seasons = load_universe_from(path)
        _, close_idx = split_by_market_phase(panel)

        for book in targets:
            _, phat_raw, phat_cal, odds, book_mask, y_sub, _ = fit_leave_out(
                panel, P, awake, y, close_idx, book, match_ids
            )
            bank_this_book = {}
            for stage, p_hat in [("raw", phat_raw), ("calibrated", phat_cal)]:
                growth, bet_placed = simulate_kelly_betting(p_hat, odds, y_sub, book_mask)
                row, bankroll = score_bet(growth, bet_placed, book_mask,
                                           {"bookmaker": book, "stage": stage})
                row["normalization"] = norm_label
                all_rows.append(row)
                curves_by_target[book][(norm_label, stage)] = bankroll
                bank_this_book[stage] = row["final_bankroll"]
            print(f"  {book:8s} raw: final_bankroll={bank_this_book['raw']:.4f}  |  "
                  f"calibrated: final_bankroll={bank_this_book['calibrated']:.4f}")

    table = pd.DataFrame(all_rows)
    table = table[["normalization", "bookmaker", "stage", "n_bets", "active_rounds",
                    "final_bankroll", "mean_log_growth_per_bet", "ci_low", "ci_high",
                    "p_value", "significant_95"]]
    out_path = os.path.join(OUT_DIR, "value_betting_closing_trained_table.csv")
    table.to_csv(out_path, index=False)
    pd.set_option("display.width", 160)
    print(f"\n=== Closing-on-closing Kelly (OGD) -- saved to {out_path} ===\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    plot_multi_target_bankrolls(
        curves_by_target, targets,
        style={
            ("Basic", "raw"): dict(color="#898781", linestyle="--", linewidth=1.4),
            ("Basic", "calibrated"): dict(color="#0b0b0b", linestyle="--", linewidth=1.8),
            ("Shin", "raw"): dict(color="#eb6834", linestyle="-", linewidth=1.4),
            ("Shin", "calibrated"): dict(color="#2a78d6", linestyle="-", linewidth=1.8),
        },
        suptitle="Closing-on-closing Kelly: OGD trained on closing-only panel, bet vs. each closing bookmaker",
        out_path=os.path.join(OUT_DIR, "value_betting_closing_trained.png"),
    )
    return table


def plot_multi_target_bankrolls(curves_by_target, targets, style, suptitle, out_path):
    """Shared plot shape for both run_closing_trained_both_norms and
    run_shin_vs_basic: one panel per target, several styled bankroll curves
    each, log-scale."""
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5), facecolor="#fcfcfb")
    if n == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        ax.set_facecolor("#fcfcfb")
        ax.axhline(1.0, color="#c3c2b7", linewidth=1.2, linestyle=":", zorder=1)
        for key, bankroll in curves_by_target[target].items():
            label = " (".join(str(k) for k in key) + ")" if isinstance(key, tuple) and len(key) > 1 else str(key)
            ax.plot(np.arange(1, len(bankroll) + 1), bankroll, label=label, zorder=2, **style[key])
        ax.set_yscale("log")
        ax.set_xlabel("Bet number", color="#52514e")
        ax.set_title(str(target), color="#0b0b0b", fontsize=11, pad=10)
        _style_axes(ax)
        legend = ax.legend(frameon=False, loc="best", fontsize=7.5)
        for text in legend.get_texts():
            text.set_color("#0b0b0b")
    axes[0].set_ylabel("Bankroll (log scale, start = 1.0)", color="#52514e")
    fig.suptitle(suptitle, color="#0b0b0b", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")


# =====================================================================
# 4. quantifying the BW look-ahead leak: honest vs. circular vs. leaky
# =====================================================================

def run_bw_leakage_check(panel, P, awake, y, open_idx, match_ids, original_table):
    """Honest (opening-only, BW excluded -- non-leaky, non-circular) vs.
    circular (opening-only, BW INCLUDED -- a sanity check, not a valid edge
    estimate: BW's own price is literally one of the mixture's inputs) vs.
    the original leaky number from run_original_two (full panel minus BW,
    which still includes every closing bookmaker). Produces
    results/opening_value_betting_corrected.csv, .png."""
    print(f"\n{'#' * 70}\n4. BW look-ahead leak: honest vs. circular vs. leaky\n{'#' * 70}")
    bw_col = panel.index(HELD_OUT_OPENING_BOOKMAKER)
    variants = [
        ("honest", "honest (12, opening-only, BW excluded)", open_idx, True),
        ("circular", "circular (13, opening-only, BW INCLUDED)", open_idx, False),
    ]

    tables, curves_by_panel = [], {}
    for curve_key, panel_label, indices, exclude_target in variants:
        _, phat_raw, phat_cal, odds, book_mask, y_sub, best_c = fit_leave_out(
            panel, P, awake, y, indices, HELD_OUT_OPENING_BOOKMAKER, match_ids, exclude_target
        )
        n_sub = len(indices) - (1 if exclude_target else 0)
        print(f"\n[{curve_key}] panel: {n_sub} bookmakers, "
              f"calibration learning-rate scale chosen on train: {best_c}")
        rows, curves = [], {}
        for stage, p_hat in [("raw", phat_raw), ("calibrated", phat_cal)]:
            growth, bet_placed = simulate_kelly_betting(p_hat, odds, y_sub, book_mask)
            row, bankroll = score_bet(growth, bet_placed, book_mask,
                                       {"held_out_bookmaker": HELD_OUT_OPENING_BOOKMAKER,
                                        "panel": panel_label, "stage": stage,
                                        "n_panel_bookmakers": n_sub})
            rows.append(row)
            curves[stage] = bankroll
            print(f"  [{stage:10s}] bets={row['n_bets']:5d} ({row['bet_rate_pct']:.1f}% of "
                  f"{row['active_rounds']} active rounds)  final_bankroll={row['final_bankroll']:.4f}  "
                  f"mean_log_growth/bet={row['mean_log_growth_per_bet']:+.5f}  p={row['p_value']}")
        tables.append(pd.DataFrame(rows))
        curves_by_panel[curve_key] = curves

    original_bw = original_table[original_table["held_out_bookmaker"] == HELD_OUT_OPENING_BOOKMAKER].copy()
    original_bw.insert(1, "panel", "leaky (25, incl. all closing)")
    original_bw.insert(3, "n_panel_bookmakers", 25)

    combined = pd.concat(tables + [original_bw], ignore_index=True, sort=False)
    combined = combined.sort_values(["stage", "panel"])
    out_path = os.path.join(OUT_DIR, "opening_value_betting_corrected.csv")
    combined.to_csv(out_path, index=False)
    pd.set_option("display.width", 160)
    print(f"\n=== Three panel variants, vs. {HELD_OUT_OPENING_BOOKMAKER} -- saved to {out_path} ===\n")
    print(combined[["panel", "stage", "n_panel_bookmakers", "n_bets", "final_bankroll",
                     "mean_log_growth_per_bet", "p_value", "significant_95"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    plot_corrected_vs_original(curves_by_panel, os.path.join(OUT_DIR, "opening_value_betting_corrected.png"))
    return combined


def plot_corrected_vs_original(curves_by_panel, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axhline(1.0, color="#898781", linewidth=1.2, linestyle="--", zorder=1)
    style = {
        ("honest", "raw"): dict(color="#eb6834", linestyle="-", linewidth=1.8),
        ("honest", "calibrated"): dict(color="#2a78d6", linestyle="-", linewidth=1.8),
        ("circular", "raw"): dict(color="#eb6834", linestyle="--", linewidth=1.4),
        ("circular", "calibrated"): dict(color="#2a78d6", linestyle="--", linewidth=1.4),
    }
    for panel_kind, table in curves_by_panel.items():
        for stage, bankroll in table.items():
            ax.plot(np.arange(1, len(bankroll) + 1), bankroll,
                     label=f"{panel_kind} ({stage})", **style[(panel_kind, stage)])
    ax.set_yscale("log")
    ax.set_xlabel("Bet number", color="#52514e")
    ax.set_ylabel("Bankroll (log scale, start = 1.0)", color="#52514e")
    ax.set_title(f"Kelly simulation vs. {HELD_OUT_OPENING_BOOKMAKER}: honest leave-one-out (solid) "
                  f"vs. circular full-opening-panel (dashed)", color="#0b0b0b", fontsize=11, pad=12)
    _style_axes(ax)
    legend = ax.legend(frameon=False, loc="best", fontsize=8.5)
    for text in legend.get_texts():
        text.set_color("#0b0b0b")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


# =====================================================================
# 5. full training-panel sweep: 2 normalizations x 3 panel types x every
#    applicable target = 108 rows
# =====================================================================

def run_training_panel_sweep(match_ids):
    """closing-trained -> closing targets only; mixed-trained (full panel
    minus target) -> all targets (KNOWN LEAKY for opening targets, flagged
    via `leakage_risk`, kept as the naive/uncorrected comparison point);
    opening-trained -> all targets (always valid: using older
    opening-time-available information to bet a later closing price is
    never a look-ahead problem, only the reverse is). Produces
    results/value_betting_training_panel_sweep.csv."""
    print(f"\n{'#' * 70}\n5. Full training-panel sweep (108 rows)\n{'#' * 70}")
    books = select_bookmakers()
    closing_books = [b for b in books if b.endswith("C")]
    print(f"Targets: {len(books)} total ({books}), {len(closing_books)} closing-side ({closing_books})\n")

    all_rows = []
    for norm_label, path in [("Basic", os.path.join(DATA_DIR, "odds_long.csv")),
                              ("Shin", os.path.join(DATA_DIR, "odds_long_shin.csv"))]:
        print(f"\n{'=' * 70}\n{norm_label} normalization\n{'=' * 70}")
        panel, P, awake, y, dates, seasons = load_universe_from(path)
        open_idx, close_idx = split_by_market_phase(panel)
        N = len(panel)

        panel_types = [
            ("closing-trained", close_idx, closing_books, lambda phase: False, True),
            ("mixed-trained", list(range(N)), books, lambda phase: phase == "opening", False),
            ("opening-trained", open_idx, books, lambda phase: False, True),
        ]
        for panel_type, indices, targets, leakage_fn, apply_reduction in panel_types:
            for book in targets:
                print(f"  [{panel_type}] -> {book}")
                phase = "closing" if book.endswith("C") else "opening"
                _, phat_raw, phat_cal, odds, book_mask, y_sub, _ = fit_leave_out(
                    panel, P, awake, y, indices, book, match_ids, apply_reduction=apply_reduction
                )
                for stage, p_hat in [("raw", phat_raw), ("calibrated", phat_cal)]:
                    growth, bet_placed = simulate_kelly_betting(p_hat, odds, y_sub, book_mask)
                    row, _ = score_bet(growth, bet_placed, book_mask, {
                        "normalization": norm_label, "panel_type": panel_type, "target_phase": phase,
                        "leakage_risk": leakage_fn(phase), "bookmaker": book, "stage": stage,
                    })
                    all_rows.append(row)

    table = pd.DataFrame(all_rows)
    table = table[["normalization", "panel_type", "target_phase", "leakage_risk", "bookmaker", "stage",
                    "n_bets", "active_rounds", "final_bankroll", "mean_log_growth_per_bet",
                    "ci_low", "ci_high", "p_value", "significant_95"]]
    out_path = os.path.join(OUT_DIR, "value_betting_training_panel_sweep.csv")
    table.to_csv(out_path, index=False)
    pd.set_option("display.width", 200)
    print(f"\n=== Full sweep: {len(table)} rows -- saved to {out_path} ===\n")

    sig = table[(table["stage"] == "calibrated") & (table["significant_95"])].sort_values("mean_log_growth_per_bet")
    print(f"=== Significant (calibrated) results: {len(sig)} of {len(table[table['stage'] == 'calibrated'])} ===\n")
    print(sig[["normalization", "panel_type", "bookmaker", "target_phase", "leakage_risk",
               "final_bankroll", "mean_log_growth_per_bet", "p_value"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    n_leaky_sig = len(sig[sig["leakage_risk"]])
    print(f"\n({n_leaky_sig} of those {len(sig)} significant rows carry the known mixed-trained/opening-target "
          f"leakage risk -- interpret with the same caution as the original BW result.)")
    return table


# =====================================================================
# 6. Shin vs. Basic value betting, for the 2 WORST algorithms
# =====================================================================

def run_shin_vs_basic(match_ids):
    """Does Shin normalization change the economics for FTRL and Uniform
    average (the two worst of the four mixture algorithms by calibrated
    log-loss -- deliberately not OGD, which experiments 1-5 already cover)?
    The odds actually bet against are identical in both datasets (de-vig
    can't change payouts), so any profit difference comes purely from
    p_hat. Produces results/shin_vs_basic_value_betting_table.csv,
    _significance.csv, .png."""
    print(f"\n{'#' * 70}\n6. Shin vs. Basic value betting, worst 2 algorithms\n{'#' * 70}")

    universes = {}
    for label, path in [("Basic", os.path.join(DATA_DIR, "odds_long.csv")),
                         ("Shin", os.path.join(DATA_DIR, "odds_long_shin.csv"))]:
        panel, P, awake, y, dates, seasons = load_universe_from(path)
        universes[label] = {"panel": panel, "P": P, "awake": awake, "y": y}
    y = universes["Basic"]["y"]
    assert np.array_equal(y, universes["Shin"]["y"]), "Basic and Shin disagree on match outcomes"

    rows = []
    curves = {}
    full_log_growth = {}  # (algo, book, normalization, stage) -> per-active-round log-growth

    for book in HELD_OUT_BOOKMAKERS:
        odds = load_decimal_odds(book, match_ids)
        book_mask = universes["Basic"]["awake"][:, universes["Basic"]["panel"].index(book)] & ~np.isnan(odds[:, 0])

        for algo in WORST_ALGORITHMS:
            print(f"\n--- {algo}, held out: {book} ---")
            for label, uni in universes.items():
                N = len(uni["panel"])
                keep_cols = [k for k, b in enumerate(uni["panel"]) if b != book]
                sub_panel = [uni["panel"][k] for k in keep_cols]
                P_sub, awake_sub = uni["P"][:, keep_cols, :], uni["awake"][:, keep_cols]

                if algo == "FTRL":
                    loss_sub = per_round_expert_loss(P_sub, y)
                    W = run_ftrl_sleeping(loss_sub, awake_sub, len(sub_panel))
                else:  # Uniform average
                    # safe divide: dropping the target column can leave a round
                    # with NOBODY awake, and a bare 0/0 here silently produced
                    # NaN probabilities -> NaN calibration loss -> best_c=None
                    denom = awake_sub.sum(axis=1, keepdims=True)
                    W = np.divide(awake_sub, denom,
                                  out=np.zeros_like(awake_sub, dtype=float),
                                  where=denom > 0)
                phat_raw = np.einsum("tn,tnk->tk", W, P_sub)

                alive = awake_sub.any(axis=1)          # rounds this subset can predict at all
                _, cal_sub, _, best_c = calibrate_full_history(phat_raw, y, alive)
                phat_cal = np.zeros_like(phat_raw)
                phat_cal[alive] = cal_sub
                bet_mask = book_mask & alive           # never bet a round with no prediction

                for stage, p_hat in [("raw", phat_raw), ("calibrated", phat_cal)]:
                    growth, bet_placed = simulate_kelly_betting(p_hat, odds, y, bet_mask)
                    row, bankroll = score_bet(growth, bet_placed, bet_mask, {
                        "algorithm": algo, "held_out_bookmaker": book, "normalization": label, "stage": stage,
                    })
                    row["profit_pct"] = (row["final_bankroll"] - 1.0) * 100 if pd.notna(row["final_bankroll"]) else np.nan
                    rows.append(row)
                    curves[(algo, book, label, stage)] = bankroll
                    full_log_growth[(algo, book, label, stage)] = np.log(growth[book_mask])  # 0 on no-bet rounds

                    print(f"  [{label:5s} {stage:10s}] bets={row['n_bets']:5d} ({row['bet_rate_pct']:.1f}% of "
                          f"{row['active_rounds']} active rounds)  profit={row['profit_pct']:+.2f}%  "
                          f"mean_log_growth/bet={row['mean_log_growth_per_bet']:+.5f}")

    table = pd.DataFrame(rows)
    table = table[["algorithm", "held_out_bookmaker", "normalization", "stage",
                    "active_rounds", "n_bets", "bet_rate_pct", "final_bankroll", "profit_pct",
                    "mean_log_growth_per_bet", "ci_low", "ci_high", "p_value", "significant_95"]]

    sig_rows = []
    for book in HELD_OUT_BOOKMAKERS:
        for algo in WORST_ALGORITHMS:
            for stage in ["raw", "calibrated"]:
                a = full_log_growth[(algo, book, "Shin", stage)]
                b = full_log_growth[(algo, book, "Basic", stage)]
                res = moving_block_bootstrap_test(a - b)
                sig_rows.append({"algorithm": algo, "held_out_bookmaker": book, "stage": stage,
                                  "kind": "shin_vs_basic", **res})
    sig_table = pd.DataFrame(sig_rows)

    out_path = os.path.join(OUT_DIR, "shin_vs_basic_value_betting_table.csv")
    table.to_csv(out_path, index=False)
    sig_path = os.path.join(OUT_DIR, "shin_vs_basic_value_betting_significance.csv")
    sig_table.to_csv(sig_path, index=False)
    pd.set_option("display.width", 160)
    print(f"\n=== Shin vs. Basic value betting -- worst 2 algorithms -- saved to {out_path} ===\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n=== Shin vs. Basic significance, log-growth per active round "
          f"(negative mean_diff = Shin has WORSE growth; positive = Shin BETTER) -- saved to {sig_path} ===\n")
    for _, r in sig_table.iterrows():
        flag = "SIGNIFICANT" if r["significant_95"] else "not significant"
        print(f"{r['algorithm']:16s} {r['held_out_bookmaker']:6s} {r['stage']:11s} n={int(r['n']):6d}  "
              f"mean_diff={r['mean_diff']:+.5f}  95% CI=[{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]  "
              f"p={r['p_value']:.4f}  {flag}")

    panels = sorted({(algo, book) for (algo, book, _, _) in curves})
    curves_by_target = {(algo, book): {} for (algo, book) in panels}
    for (algo, book, norm, stage), bankroll in curves.items():
        curves_by_target[(algo, book)][(norm, stage)] = bankroll
    plot_multi_target_bankrolls(
        curves_by_target, panels,
        style={
            ("Basic", "raw"): dict(color="#898781", linestyle="--", linewidth=1.4),
            ("Basic", "calibrated"): dict(color="#0b0b0b", linestyle="--", linewidth=1.8),
            ("Shin", "raw"): dict(color="#eb6834", linestyle="-", linewidth=1.4),
            ("Shin", "calibrated"): dict(color="#2a78d6", linestyle="-", linewidth=1.8),
        },
        suptitle="Quarter-Kelly bankroll: Basic vs. Shin normalization, worst 2 algorithms",
        out_path=os.path.join(OUT_DIR, "shin_vs_basic_value_betting.png"),
    )
    return table, sig_table


# =====================================================================

def main():
    panel, P, awake, y, dates, seasons = load_universe_from(os.path.join(DATA_DIR, "odds_long.csv"))
    open_idx, close_idx = split_by_market_phase(panel)
    match_ids = load_match_order()
    assert len(match_ids) == P.shape[0], "match_ids ordering must line up with load_universe_from()"

    original_table = run_original_two(panel, P, awake, y, match_ids)
    run_all_bookmakers_sweep(panel, P, awake, y, open_idx, close_idx, match_ids)
    run_closing_trained_both_norms(match_ids)
    run_bw_leakage_check(panel, P, awake, y, open_idx, match_ids, original_table)
    run_training_panel_sweep(match_ids)
    run_shin_vs_basic(match_ids)


if __name__ == "__main__":
    main()
