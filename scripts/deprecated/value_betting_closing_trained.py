"""
Standalone "closing-on-closing" Kelly betting comparison: OGD trained on
ONLY the other closing-side bookmakers (leave-one-out within the closing
subset), bet against every closing-side target -- isolated into its own
file (with bankroll trajectory plots) rather than buried as one slice of
the larger value_betting_training_panel_sweep.py sweep. Run for both
de-vig normalizations (Basic, Shin).

Causally valid, no look-ahead risk: every closing bookmaker's price settles
at roughly the same time, near kickoff, regardless of when each one opened
-- so using OTHER closing bookmakers as the training panel for a
closing-side target isn't a look-ahead artifact (contrast with an
opening-side target, where the training panel must exclude closing
information entirely -- see opening_closing_quality_transfer.py).

Produces:
  - results/value_betting_closing_trained_table.csv
  - results/value_betting_closing_trained.png  (bankroll trajectories, log
    scale, one panel per closing-side target, Basic vs. Shin x raw vs.
    calibrated)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import OUT_DIR, DATA_DIR, per_round_expert_loss, run_ogd_sleeping
from final_ranking import calibrate_full_history
from significance_test import moving_block_bootstrap_test
from opening_vs_closing import split_by_market_phase
from value_betting import load_match_order, load_decimal_odds, simulate_kelly_betting, STARTING_BANKROLL
from value_betting_all_bookmakers import select_bookmakers
from shin_vs_basic_comparison import load_universe_from


def run_closing_on_closing(panel, P, awake, y, book, close_idx, match_ids):
    """OGD fitted on ONLY the other closing-side bookmakers (excludes
    `book` itself, and excludes every opening-side bookmaker entirely),
    Kelly-bet against `book`'s real closing odds. Returns (rows, curves) --
    rows for the summary table, curves for the bankroll-trajectory plot."""
    book_col = panel.index(book)
    keep_cols = [k for k in close_idx if k != book_col]
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
    _, phat_cal, _, best_c = calibrate_full_history(phat_raw, y_sub, all_true)

    rows, curves = [], {}
    for stage, p_hat in [("raw", phat_raw), ("calibrated", phat_cal)]:
        growth, bet_placed = simulate_kelly_betting(p_hat, odds, y_sub, book_mask)
        bankroll = STARTING_BANKROLL * np.cumprod(growth[book_mask])
        curves[stage] = bankroll

        log_growth_per_bet = np.log(growth[bet_placed])
        n_bets = int(bet_placed.sum())
        n_active = int(book_mask.sum())
        if n_bets >= 30:
            res = moving_block_bootstrap_test(log_growth_per_bet)
        else:
            res = {"mean_diff": log_growth_per_bet.mean() if n_bets else np.nan,
                   "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "significant_95": False}

        rows.append({
            "bookmaker": book, "n_panel_bookmakers": N_sub, "stage": stage,
            "active_rounds": n_active, "n_bets": n_bets, "bet_rate_pct": 100 * n_bets / max(1, n_active),
            "final_bankroll": bankroll[-1] if len(bankroll) else np.nan,
            "mean_log_growth_per_bet": res["mean_diff"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
            "p_value": res["p_value"], "significant_95": res["significant_95"],
        })
    return rows, curves


def plot_bankrolls(curves_by_target, targets, out_path):
    """Saves value_betting_closing_trained.png: one panel per closing-side
    target, 4 lines each (Basic/Shin x raw/calibrated), log-scale bankroll
    -- same convention as value_betting.plot_bankrolls / the other
    multi-panel bankroll plots built this session."""
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5), facecolor="#fcfcfb")
    if n == 1:
        axes = [axes]

    style = {
        ("Basic", "raw"): dict(color="#898781", linestyle="--", linewidth=1.4),
        ("Basic", "calibrated"): dict(color="#0b0b0b", linestyle="--", linewidth=1.8),
        ("Shin", "raw"): dict(color="#eb6834", linestyle="-", linewidth=1.4),
        ("Shin", "calibrated"): dict(color="#2a78d6", linestyle="-", linewidth=1.8),
    }

    for ax, target in zip(axes, targets):
        ax.set_facecolor("#fcfcfb")
        ax.axhline(1.0, color="#c3c2b7", linewidth=1.2, linestyle=":", zorder=1)
        for (norm, stage), bankroll in curves_by_target[target].items():
            ax.plot(np.arange(1, len(bankroll) + 1), bankroll,
                     label=f"{norm} ({stage})", zorder=2, **style[(norm, stage)])
        ax.set_yscale("log")
        ax.set_xlabel("Bet number", color="#52514e")
        ax.set_title(target, color="#0b0b0b", fontsize=11, pad=10)
        ax.grid(True, which="both", color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors="#898781")
        legend = ax.legend(frameon=False, loc="best", fontsize=7.5)
        for text in legend.get_texts():
            text.set_color("#0b0b0b")

    axes[0].set_ylabel("Bankroll (log scale, start = 1.0)", color="#52514e")
    fig.suptitle("Closing-on-closing Kelly: OGD trained on closing-only panel, bet vs. each closing bookmaker",
                  color="#0b0b0b", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")


def main():
    books = select_bookmakers()
    targets = [b for b in books if b.endswith("C")]
    print(f"Closing-side targets: {targets}\n")

    match_ids = load_match_order()

    all_rows = []
    curves_by_target = {t: {} for t in targets}

    for norm_label, path in [("Basic", os.path.join(DATA_DIR, "odds_long.csv")),
                              ("Shin", os.path.join(DATA_DIR, "odds_long_shin.csv"))]:
        print(f"\n{'=' * 60}\n{norm_label} normalization\n{'=' * 60}")
        panel, P, awake, y, dates, seasons = load_universe_from(path)
        _, close_idx = split_by_market_phase(panel)

        for book in targets:
            rows, curves = run_closing_on_closing(panel, P, awake, y, book, close_idx, match_ids)
            for r in rows:
                r["normalization"] = norm_label
                all_rows.append(r)
                curves_by_target[book][(norm_label, r["stage"])] = curves[r["stage"]]
            print(f"  {book:8s} raw: final_bankroll={rows[0]['final_bankroll']:.4f} "
                  f"p={rows[0]['p_value']}  |  calibrated: final_bankroll={rows[1]['final_bankroll']:.4f} "
                  f"p={rows[1]['p_value']}")

    table = pd.DataFrame(all_rows)
    table = table[["normalization", "bookmaker", "stage", "n_panel_bookmakers", "n_bets", "active_rounds",
                    "final_bankroll", "mean_log_growth_per_bet", "ci_low", "ci_high", "p_value", "significant_95"]]
    out_path = os.path.join(OUT_DIR, "value_betting_closing_trained_table.csv")
    table.to_csv(out_path, index=False)

    pd.set_option("display.width", 160)
    print(f"\n=== Closing-on-closing Kelly (OGD) -- saved to {out_path} ===\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    plot_bankrolls(curves_by_target, targets, os.path.join(OUT_DIR, "value_betting_closing_trained.png"))


if __name__ == "__main__":
    main()
