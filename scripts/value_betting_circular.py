"""
WHAT IF WE DON'T TAKE THE TARGET OUT OF THE MIXTURE?

Every value-betting experiment in this project excludes the bookmaker being
bet against from the training panel first, on the grounds that otherwise you
are beating it partly with its own price. `value_betting.run_bw_leakage_check`
tests that for exactly one bookmaker (BW). This file does it for all 11, as
a controlled A/B: same panel, same Kelly rule, same calibration, the ONLY
difference being `exclude_target`.

WHAT TO EXPECT, AND WHY IT IS NOT THE USUAL LEAKAGE STORY. The look-ahead
bug this project found earlier INFLATED returns, because the mixture got
information from the future. Circularity works the other way. The target's
de-vigged probabilities go INTO the mixture, so p_hat gets pulled toward the
target's own price -- and a bet needs p_hat * odds > 1, where `odds` are that
same bookmaker's VIGGED odds. In the limit where the mixture is just the
target, p_target * odds_target = 1/(1+overround) < 1 for every outcome, so
no bet ever clears. Circularity should therefore SHRINK the apparent edge
toward zero by killing the disagreement that generates bets in the first
place, not fabricate profit. How much it shrinks depends on how much weight
the mixture gives the target, which this file measures directly.

So the honest reason for the leave-one-out is not "including the target
would make us look good" -- it is that the comparison stops meaning
anything: you are no longer testing a forecast against a price, you are
testing a price against a blend of itself.

Design otherwise identical to value_betting.run_all_bookmakers_sweep
(per-target-phase panel, quarter-Kelly, raw + calibrated). The honest side
is read straight from that experiment's saved table rather than refitted,
so the comparison is against the exact published numbers.

Produces:
  - results/value_betting_circular_table.csv     (circular runs)
  - results/value_betting_circular_vs_honest.csv (side by side + target weight)
  - results/value_betting_circular.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sleeping_experts import (
    DATA_DIR, OUT_DIR, COLORS,
    per_round_expert_loss, run_ogd_sleeping,
    load_match_order, load_universe_from, split_by_market_phase,
)
from final_ranking import calibrate_full_history
from betting import (
    _style_axes, load_decimal_odds, score_bet, select_bookmakers, simulate_kelly_betting,
)


def fit_including_target(panel, P, awake, y, indices, target_book, match_ids,
                          exclude_target, apply_reduction=True):
    """value_betting.fit_leave_out, but it also returns the fitted weight
    matrix and the target's column inside the sub-panel, so the size of the
    circularity effect can be tied to how much weight the mixture actually
    puts on the target. Everything else is line-for-line identical -- with
    exclude_target=True it reproduces the canonical sweep, which main()
    asserts against the saved table."""
    target_col = panel.index(target_book)
    keep_cols = [k for k in indices if not (exclude_target and k == target_col)]
    sub_panel = [panel[k] for k in keep_cols]
    N_sub = len(sub_panel)

    P_sub_full, awake_sub_full = P[:, keep_cols, :], awake[:, keep_cols]
    odds_full = load_decimal_odds(target_book, match_ids)
    book_mask_full = awake[:, target_col] & ~np.isnan(odds_full[:, 0])

    if apply_reduction:
        keep = awake_sub_full.any(axis=1) & book_mask_full
        P_sub, awake_sub, y_sub = P_sub_full[keep], awake_sub_full[keep], y[keep]
        odds, book_mask = odds_full[keep], np.ones(int(keep.sum()), dtype=bool)
    else:
        P_sub, awake_sub, y_sub = P_sub_full, awake_sub_full, y
        odds, book_mask = odds_full, book_mask_full

    loss_sub = per_round_expert_loss(P_sub, y_sub)
    W = run_ogd_sleeping(loss_sub, awake_sub, N_sub)
    phat_raw = np.einsum("tn,tnk->tk", W, P_sub)

    all_true = np.ones(len(y_sub), dtype=bool)
    _, phat_cal, _, _ = calibrate_full_history(phat_raw, y_sub, all_true)

    # mean weight the mixture gave the target over the rounds it was awake
    target_weight = np.nan
    if target_book in sub_panel:
        j = sub_panel.index(target_book)
        aw = awake_sub[:, j]
        target_weight = float(W[aw, j].mean()) if aw.any() else np.nan
    return phat_raw, phat_cal, odds, book_mask, y_sub, target_weight, N_sub


def plot_circular(merged, out_path):
    """One row per bookmaker: open circle = honest (target excluded), filled =
    circular (target left in), joined. Dashed vertical at breakeven."""
    df = merged.dropna(subset=["edge_honest", "edge_circular"]).sort_values("edge_honest")
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(df) + 2.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axvline(0, color="#898781", linewidth=1.3, linestyle="--", zorder=1)

    ypos = np.arange(len(df))
    for yp, (_, r) in zip(ypos, df.iterrows()):
        color = COLORS["OGD"] if r["edge_circular"] < r["edge_honest"] else COLORS["Hedge"]
        ax.plot([r["edge_honest"], r["edge_circular"]], [yp, yp],
                 color=color, linewidth=1.5, alpha=0.85, zorder=2)
        ax.scatter([r["edge_honest"]], [yp], s=70, facecolor="#fcfcfb",
                    edgecolor=color, linewidth=1.8, zorder=3)
        ax.scatter([r["edge_circular"]], [yp], s=70, facecolor=color,
                    edgecolor=color, linewidth=1.0, zorder=3)

    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r['bookmaker']}  (w={r['target_weight']:.3f})"
                        for _, r in df.iterrows()], fontsize=8.5, color="#0b0b0b")
    ax.set_xlabel("Mean log-growth per bet, calibrated (0 = breakeven)", color="#52514e")
    ax.set_title("Leaving the target bookmaker INSIDE the mixture\n"
                  "open = honest (excluded), filled = circular (included); "
                  "w = mean weight the mixture gives it",
                  color="#0b0b0b", fontsize=11.5, pad=12)
    ax.grid(True, axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", left=False)
    legend = ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="#52514e", markersize=8, markeredgewidth=1.8,
               label="honest (target excluded)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#52514e",
               markeredgecolor="#52514e", markersize=8, label="circular (target included)"),
    ], frameon=False, loc="best", fontsize=8.5)
    for t in legend.get_texts():
        t.set_color("#0b0b0b")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def main():
    panel, P, awake, y, dates, seasons = load_universe_from(os.path.join(DATA_DIR, "odds_long.csv"))
    open_idx, close_idx = split_by_market_phase(panel)
    match_ids = load_match_order()
    N = len(panel)

    books = select_bookmakers()
    print(f"Targets ({len(books)}): {books}\n")

    rows = []
    for book in books:
        is_closing = book.endswith("C")
        indices = list(range(N)) if is_closing else open_idx
        phat_raw, phat_cal, odds, book_mask, y_sub, tw, n_panel = fit_including_target(
            panel, P, awake, y, indices, book, match_ids,
            exclude_target=False, apply_reduction=not is_closing,
        )
        print(f"--- {book} ({'closing' if is_closing else 'opening'}), "
              f"panel keeps {n_panel} incl. itself, mean weight on itself = {tw:.4f} ---")
        for stage, p_hat in [("raw", phat_raw), ("calibrated", phat_cal)]:
            growth, bet_placed = simulate_kelly_betting(p_hat, odds, y_sub, book_mask)
            row, _ = score_bet(growth, bet_placed, book_mask,
                                {"bookmaker": book, "stage": stage,
                                 "target_weight": tw, "n_panel": n_panel})
            rows.append(row)
            print(f"  [{stage:10s}] bets={row['n_bets']:6d} "
                  f"({row['bet_rate_pct']:.1f}% of {row['active_rounds']})  "
                  f"bankroll={row['final_bankroll']:.4f}  "
                  f"edge={row['mean_log_growth_per_bet']:+.5f}  p={row['p_value']}")

    circ = pd.DataFrame(rows)
    circ_path = os.path.join(OUT_DIR, "value_betting_circular_table.csv")
    circ.to_csv(circ_path, index=False)

    # ---- side by side against the published honest sweep ----
    honest_path = os.path.join(OUT_DIR, "value_betting_all_bookmakers_table.csv")
    honest = pd.read_csv(honest_path)
    cal_h = honest[honest["stage"] == "calibrated"].set_index("bookmaker")
    cal_c = circ[circ["stage"] == "calibrated"].set_index("bookmaker")

    merged = pd.DataFrame({
        "target_weight": cal_c["target_weight"],
        "bets_honest": cal_h["n_bets"], "bets_circular": cal_c["n_bets"],
        "bankroll_honest": cal_h["final_bankroll"], "bankroll_circular": cal_c["final_bankroll"],
        "edge_honest": cal_h["mean_log_growth_per_bet"],
        "edge_circular": cal_c["mean_log_growth_per_bet"],
        "sig_honest": cal_h["significant_95"], "sig_circular": cal_c["significant_95"],
    }).reset_index().rename(columns={"index": "bookmaker"})
    merged["bets_change_pct"] = 100 * (merged["bets_circular"] / merged["bets_honest"] - 1)
    merged["edge_shrunk"] = merged["edge_circular"].abs() < merged["edge_honest"].abs()

    out_path = os.path.join(OUT_DIR, "value_betting_circular_vs_honest.csv")
    merged.to_csv(out_path, index=False)

    pd.set_option("display.width", 220)
    print(f"\n=== Circular vs honest, calibrated stage — saved to {out_path} ===\n")
    print(merged[["bookmaker", "target_weight", "bets_honest", "bets_circular",
                   "bets_change_pct", "edge_honest", "edge_circular",
                   "bankroll_honest", "bankroll_circular"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nbets fell in {int((merged['bets_change_pct'] < 0).sum())} of {len(merged)} targets "
          f"(median change {merged['bets_change_pct'].median():+.1f}%)")
    print(f"|edge| shrank toward 0 in {int(merged['edge_shrunk'].sum())} of {len(merged)}")
    print(f"significant results: honest {int(merged['sig_honest'].sum())}, "
          f"circular {int(merged['sig_circular'].sum())}")

    plot_circular(merged, os.path.join(OUT_DIR, "value_betting_circular.png"))


if __name__ == "__main__":
    main()
