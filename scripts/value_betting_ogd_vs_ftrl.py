"""
OGD vs. FTRL, HEAD-TO-HEAD, IN THE VALUE-BETTING SIMULATION.

Why this file exists: value_betting.py's experiments 1-5 use ONLY OGD (it's
the algorithm significance_test.py singled out as best, so the only one
"worth putting money behind"), and its experiment 6 uses only the two WORST
algorithms (FTRL, Uniform average) to ask a de-vig question. The two
therefore overlap on exactly one methodologically valid cell (B365C, Basic
normalization), and even there the numbers are two INDEPENDENT "is this
edge different from zero?" tests -- there is no paired test anywhere in the
project of the form "is OGD's betting performance different from FTRL's".
That is the question this file answers.

Kept as its own file rather than a 7th experiment inside value_betting.py,
for the same reason value_betting_online_kelly.py is separate: it varies the
MIXTURE ALGORITHM while holding the training panel, normalization and
staking rule fixed, which is a different axis from every experiment there.
It carries its own small fit routine (a copy of value_betting.fit_leave_out
with the algorithm as a parameter) instead of adding a parameter to that
function, which all 6 of value_betting.py's experiments depend on.

DESIGN -- identical to value_betting.run_all_bookmakers_sweep (the project's
canonical honest sweep), so the per-algorithm rows here are directly
comparable to results/value_betting_all_bookmakers_table.csv:
  - closing-side target: standard leave-one-out (full panel minus target --
    valid, every closing price settles near kickoff);
  - opening-side target: corrected leave-one-out (opening bookmakers only,
    target excluded -- no closing-side information exists yet at
    opening-bet time; see value_betting.py's module docstring);
  - quarter-Kelly on the single highest-positive-EV outcome, raw and
    causally calibrated p_hat, over the 11 bookmakers with coverage >= 50%
    that are not single-season.

THE PAIRED TEST -- and why it is NOT "the same bets". Two mixture
algorithms produce different probability estimates, so they disagree about
WHICH rounds have positive EV and sometimes about which outcome to back.
So, unlike value_betting_online_kelly.py (where both staking rules play the
identical bet and a per-bet pairing is exact), the pairing here has to be
over ROUNDS, not bets: per-active-round log-growth, which is 0 on a round
where that algorithm chose not to bet. Same device as
value_betting.run_shin_vs_basic. A round both skip contributes an exact 0 to
the difference and only costs variance, never bias.

SIGN CONVENTION -- read carefully, it is the opposite of the project's
log-loss tables: the bootstrapped quantity is log-GROWTH (wealth), so
    diff = log-growth(OGD) - log-growth(FTRL),  POSITIVE = OGD is BETTER.
(See CLAUDE.md's sign-convention gotchas.)

Produces:
  - results/value_betting_ogd_vs_ftrl_table.csv        (one row per
      bookmaker x algorithm x stage: bets, bankroll, edge, own CI/p-value)
  - results/value_betting_ogd_vs_ftrl_significance.csv (one row per
      bookmaker x stage: the PAIRED OGD-vs-FTRL bootstrap)
  - results/value_betting_ogd_vs_ftrl.png              (dot plot: each
      bookmaker's edge under both algorithms)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sleeping_experts import (
    DATA_DIR, OUT_DIR, COLORS,
    per_round_expert_loss, run_ogd_sleeping, run_ftrl_sleeping,
    load_match_order, load_universe_from, split_by_market_phase,
)
from final_ranking import calibrate_full_history
from significance_test import moving_block_bootstrap_test
from betting import (
    _style_axes, load_decimal_odds, score_bet, select_bookmakers, simulate_kelly_betting,
)

ALGORITHMS = {"OGD": run_ogd_sleeping, "FTRL": run_ftrl_sleeping}


def fit_leave_out_algo(panel, P, awake, y, indices, target_book, match_ids, algo,
                        apply_reduction=True):
    """value_betting.fit_leave_out with the mixture algorithm as a parameter
    (that one hardcodes run_ogd_sleeping). Everything else -- the column
    subset, the target exclusion, the `apply_reduction` round filter and the
    causal calibration -- is deliberately identical, so a row produced here
    with algo="OGD" reproduces the corresponding row of
    value_betting_all_bookmakers_table.csv exactly (asserted in main())."""
    target_col = panel.index(target_book)
    keep_cols = [k for k in indices if k != target_col]
    N_sub = len(keep_cols)

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
    W = ALGORITHMS[algo](loss_sub, awake_sub, N_sub)
    phat_raw = np.einsum("tn,tnk->tk", W, P_sub)

    all_true = np.ones(len(y_sub), dtype=bool)
    _, phat_cal, _, _ = calibrate_full_history(phat_raw, y_sub, all_true)
    return phat_raw, phat_cal, odds, book_mask, y_sub


def plot_comparison(table, sig_table, out_path):
    """One row per bookmaker, two dots: OGD and FTRL mean log-growth per bet
    (calibrated stage), joined by a line. Dashed vertical at 0 = breakeven.
    A dot/position encoding rather than bars, per the project's convention
    for quantities with no meaningful zero baseline to anchor a bar to."""
    cal = table[table["stage"] == "calibrated"]
    piv = cal.pivot(index="bookmaker", columns="algorithm", values="mean_log_growth_per_bet")
    sig = sig_table[sig_table["stage"] == "calibrated"].set_index("bookmaker")
    piv = piv.reindex(sig.index.intersection(piv.index)).sort_values("OGD")

    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(piv) + 2.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axvline(0, color="#898781", linewidth=1.3, linestyle="--", zorder=1)

    ypos = np.arange(len(piv))
    for yp, (book, row) in zip(ypos, piv.iterrows()):
        ax.plot([row["FTRL"], row["OGD"]], [yp, yp], color="#c3c2b7", linewidth=1.4, zorder=2)
        ax.scatter([row["FTRL"]], [yp], s=70, color=COLORS["FTRL"], zorder=3,
                    edgecolor="#fcfcfb", linewidth=1.0)
        ax.scatter([row["OGD"]], [yp], s=70, color=COLORS["OGD"], zorder=3,
                    edgecolor="#fcfcfb", linewidth=1.0)

    labels = [f"{b}{' *' if sig.loc[b, 'significant_95'] else ''}" for b in piv.index]
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.5, color="#0b0b0b")
    ax.set_xlabel("Mean log-growth per bet, calibrated (0 = breakeven, quarter-Kelly)", color="#52514e")
    ax.set_title("Value betting: OGD vs. FTRL on the same honest leave-one-out design\n"
                  "(* = the paired OGD-vs-FTRL difference is significant at 95%)",
                  color="#0b0b0b", fontsize=11.5, pad=12)
    ax.grid(True, axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", left=False)

    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="none", color=COLORS["OGD"], markersize=8, label="OGD"),
        Line2D([0], [0], marker="o", linestyle="none", color=COLORS["FTRL"], markersize=8, label="FTRL"),
    ]
    legend = ax.legend(handles=legend_elements, frameon=False, loc="best", fontsize=8.5)
    for text in legend.get_texts():
        text.set_color("#0b0b0b")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def main():
    panel, P, awake, y, dates, seasons = load_universe_from(os.path.join(DATA_DIR, "odds_long.csv"))
    open_idx, close_idx = split_by_market_phase(panel)
    match_ids = load_match_order()
    assert len(match_ids) == P.shape[0], "match_ids ordering must line up with load_universe_from()"
    N = len(panel)

    books = select_bookmakers()
    print(f"Testing {len(books)} bookmakers, OGD vs FTRL on identical panels: {books}\n")

    rows, sig_rows = [], []
    for book in books:
        is_closing = book.endswith("C")
        indices = list(range(N)) if is_closing else open_idx
        method = "closing (standard leave-one-out)" if is_closing else "opening (corrected leave-one-out)"
        print(f"--- {book} ({'closing' if is_closing else 'opening'}) ---")

        # per-active-round log-growth, keyed by (algo, stage), for the paired test
        growth_by = {}
        for algo in ALGORITHMS:
            phat_raw, phat_cal, odds, book_mask, y_sub = fit_leave_out_algo(
                panel, P, awake, y, indices, book, match_ids, algo,
                apply_reduction=not is_closing,
            )
            for stage, p_hat in [("raw", phat_raw), ("calibrated", phat_cal)]:
                growth, bet_placed = simulate_kelly_betting(p_hat, odds, y_sub, book_mask)
                row, _ = score_bet(growth, bet_placed, book_mask,
                                    {"bookmaker": book, "algorithm": algo,
                                     "method": method, "stage": stage})
                rows.append(row)
                growth_by[(algo, stage)] = np.log(growth[book_mask])  # 0 on no-bet rounds
                print(f"  [{algo:4s} {stage:10s}] bets={row['n_bets']:5d}  "
                      f"final_bankroll={row['final_bankroll']:.4f}  "
                      f"edge={row['mean_log_growth_per_bet']:+.5f}  p={row['p_value']}")

        for stage in ["raw", "calibrated"]:
            diff = growth_by[("OGD", stage)] - growth_by[("FTRL", stage)]
            res = moving_block_bootstrap_test(diff)
            sig_rows.append({"bookmaker": book, "method": method, "stage": stage, **res})
            flag = "SIGNIFICANT" if res["significant_95"] else "not significant"
            better = "OGD" if res["mean_diff"] > 0 else "FTRL"
            print(f"  [paired {stage:10s}] mean_diff={res['mean_diff']:+.6f} "
                  f"({better} ahead)  p={res['p_value']:.4f}  {flag}")
        print()

    table = pd.DataFrame(rows)
    table = table[["bookmaker", "method", "algorithm", "stage", "active_rounds", "n_bets",
                    "bet_rate_pct", "final_bankroll", "mean_log_growth_per_bet",
                    "ci_low", "ci_high", "p_value", "significant_95"]]
    table_path = os.path.join(OUT_DIR, "value_betting_ogd_vs_ftrl_table.csv")
    table.to_csv(table_path, index=False)

    sig_table = pd.DataFrame(sig_rows)
    sig_path = os.path.join(OUT_DIR, "value_betting_ogd_vs_ftrl_significance.csv")
    sig_table.to_csv(sig_path, index=False)

    pd.set_option("display.width", 200)
    print(f"\n=== Per-algorithm results -- saved to {table_path} ===\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print(f"\n=== PAIRED OGD vs FTRL, per active round -- saved to {sig_path} ===")
    print("(POSITIVE mean_diff = OGD has the HIGHER/better log-growth -- this is a GROWTH")
    print(" table, so the sign reads the opposite way from the project's log-loss tables)\n")
    print(sig_table[["bookmaker", "stage", "n", "mean_diff", "ci_low", "ci_high",
                      "p_value", "significant_95"]]
          .to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    n_sig = int(sig_table["significant_95"].sum())
    n_ogd = int((sig_table["mean_diff"] > 0).sum())
    print(f"\nOGD ahead in {n_ogd} of {len(sig_table)} comparisons; "
          f"{n_sig} of {len(sig_table)} significant at 95%.")

    verify_against_canonical_sweep(table)

    plot_comparison(table, sig_table, os.path.join(OUT_DIR, "value_betting_ogd_vs_ftrl.png"))


def verify_against_canonical_sweep(table):
    """The OGD half of this file must reproduce
    value_betting.run_all_bookmakers_sweep exactly -- same panel, same
    exclusion, same Kelly rule, same fixed bootstrap seed. Checking it here
    (rather than trusting that the copied fit routine stayed in sync) is what
    makes the FTRL half trustworthy: if OGD reproduces, the only thing that
    differs between the two halves is the algorithm."""
    ref_path = os.path.join(OUT_DIR, "value_betting_all_bookmakers_table.csv")
    if not os.path.exists(ref_path):
        print(f"\n[verify] skipped: {ref_path} not found (run value_betting.py first)")
        return
    cols = ["active_rounds", "n_bets", "final_bankroll", "mean_log_growth_per_bet", "p_value"]
    ref = pd.read_csv(ref_path).set_index(["bookmaker", "stage"])[cols].sort_index()
    mine = (table[table["algorithm"] == "OGD"]
            .set_index(["bookmaker", "stage"])[cols].sort_index())
    try:
        pd.testing.assert_frame_equal(ref, mine, check_exact=False, rtol=1e-9)
        print(f"\n[verify] OK: the OGD rows reproduce {os.path.basename(ref_path)} exactly "
              f"({len(mine)} rows) -- the only difference between the two halves is the algorithm.")
    except AssertionError as e:
        print(f"\n[verify] MISMATCH against {os.path.basename(ref_path)} -- the fit routine here has "
              f"drifted from value_betting.fit_leave_out:\n{e}")


if __name__ == "__main__":
    main()
