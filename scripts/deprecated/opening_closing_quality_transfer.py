"""
Two checks motivated by a methodological question about value_betting.py's
leave-one-bookmaker-out design: it excludes only the exact held-out
bookmaker column (e.g. "BW") before fitting OGD, but when the held-out
bookmaker is an OPENING-odds one, the remaining panel still includes its own
CLOSING counterpart ("BWC") and every other CLOSING bookmaker FOR THE SAME
MATCH. In reality none of that closing-side information exists yet at the
moment only "BW"'s (opening) price is posted -- every bookmaker's closing
price settles near kickoff, days/weeks after most opening prices are set.
So value_betting.py's "large edge against BW" implicitly assumes access to
same-match closing-side information a real early bettor would not actually
have -- a look-ahead bias the round-level (per-MATCH, not per-quote-time)
chronology in odds_long.csv can't detect on its own.

  1. DESCRIPTIVE, no leakage risk: is a bookmaker's CLOSING-side skill
     (raw log-loss, from results_table.csv, which is already computed the
     standard causal way) predictive of its own OPENING-side skill? I.e. is
     sharpness a persistent property of the underlying odds-setter, or
     purely a function of how late the price was set? Pearson + Spearman
     correlation across the 13 opening/closing bookmaker pairs
     (split_by_market_phase(), reused from opening_vs_closing.py).

  2. CORRECTED economic test: rebuilds value_betting.py's BW test with a
     properly OPENING-ONLY leave-one-out panel -- excludes every closing-
     suffixed bookmaker entirely (not just BW's own pair) from the mixture
     used to bet against BW, since none of that information would actually
     be available at opening-bet time. Compared directly against the
     original (leaky) value_betting_table.csv row to quantify how much of
     the original edge was a look-ahead artifact vs. a real opening-market
     inefficiency. Only OGD is tested, matching value_betting.py's own
     scope (the only one of the 4 algorithms significance_test.py found to
     significantly beat uniform averaging).

Produces:
  - results/opening_closing_quality_transfer.csv   (13-pair opening vs.
    closing log-loss table + correlation)
  - results/opening_closing_quality_transfer.png    (scatter: closing
    log-loss vs. opening log-loss, one point per bookmaker pair)
  - results/opening_value_betting_corrected.csv     (honest leave-one-out
    opening-only Kelly simulation for BW, raw + calibrated, alongside the
    original leaky number for direct comparison)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import OUT_DIR, load_full_universe
from final_ranking import calibrate_full_history
from significance_test import moving_block_bootstrap_test
from opening_vs_closing import split_by_market_phase
from value_betting import (
    load_match_order, load_decimal_odds, simulate_kelly_betting,
    KELLY_MULT, STARTING_BANKROLL,
)
from sleeping_experts import per_round_expert_loss, run_ogd_sleeping

HELD_OUT_OPENING_BOOKMAKER = "BW"  # same bookmaker value_betting.py's leaky test used


# ------------------------------------------------ part 1: correlation ------

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


# ----------------------------------------- part 2: corrected value betting -

def run_opening_panel_bw_test(panel, P, awake, y, panel_indices, panel_label, match_ids):
    """Fits OGD on an arbitrary opening-side panel (`panel_indices`), then
    Kelly-bets against BW's real opening odds. Called twice from main():
      - the leave-one-out panel (12 bookmakers, BW excluded) -- the honest,
        non-leaky, non-circular test;
      - the FULL opening panel (13 bookmakers, BW included) -- this is
        "the model we trained on opening odds" as it already exists
        elsewhere in the project (OGD (opening) in
        opening_vs_closing_table.csv), which is CIRCULAR by
        value_betting.py's own stated standard (BW's own price is literally
        one of the mixture's inputs) -- included only as an upper-bound /
        sanity-check, not a valid edge estimate on its own.

    Unlike value_betting.py's own panel_excluding() (drops one column out of
    26 -- some other bookmaker is essentially always awake), restricting to
    just the ~12-13 opening bookmakers can leave rounds where NONE of them
    quoted -- load_full_universe() only guarantees >=1 awake out of the
    full 26. So this applies the same "keep rounds with >=1 awake in this
    subset" reduction build_subset() uses in opening_vs_closing.py, then
    further restricts to rounds where BW itself also has usable odds."""
    sub_panel = [panel[k] for k in panel_indices]
    N_sub = len(sub_panel)
    print(f"\n[{panel_label}] panel: {N_sub} bookmakers -- {sub_panel}")

    P_sub_full, awake_sub_full = P[:, panel_indices, :], awake[:, panel_indices]
    odds_full = load_decimal_odds(HELD_OUT_OPENING_BOOKMAKER, match_ids)
    book_mask_full = awake[:, panel.index(HELD_OUT_OPENING_BOOKMAKER)] & ~np.isnan(odds_full[:, 0])

    # a round only usable if >=1 of the subset is awake AND BW itself is bettable
    keep = awake_sub_full.any(axis=1) & book_mask_full
    P_sub, awake_sub, y_sub = P_sub_full[keep], awake_sub_full[keep], y[keep]
    odds, book_mask = odds_full[keep], np.ones(keep.sum(), dtype=bool)
    print(f"  {int(keep.sum())} usable rounds (>=1 of the {N_sub} bookmakers awake AND "
          f"{HELD_OUT_OPENING_BOOKMAKER} bettable), out of {len(y)} total")

    loss_sub = per_round_expert_loss(P_sub, y_sub)
    W_ogd = run_ogd_sleeping(loss_sub, awake_sub, N_sub)
    phat_raw = np.einsum("tn,tnk->tk", W_ogd, P_sub)

    all_true = np.ones(len(y_sub), dtype=bool)
    _, phat_cal, _, best_c = calibrate_full_history(phat_raw, y_sub, all_true)
    print(f"  calibration learning-rate scale chosen on train: {best_c}")

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
            "held_out_bookmaker": HELD_OUT_OPENING_BOOKMAKER, "panel": panel_label, "stage": stage,
            "n_panel_bookmakers": N_sub, "active_rounds": n_active, "n_bets": n_bets,
            "bet_rate_pct": 100 * n_bets / max(1, n_active),
            "final_bankroll": bankroll[-1] if len(bankroll) else np.nan,
            "mean_log_growth_per_bet": res["mean_diff"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
            "p_value": res["p_value"], "significant_95": res["significant_95"],
        })
        print(f"  [{stage:10s}] bets={n_bets:5d} ({100*n_bets/max(1,n_active):.1f}% of {n_active} active rounds)  "
              f"final_bankroll={bankroll[-1] if len(bankroll) else float('nan'):.4f}  "
              f"mean_log_growth/bet={res['mean_diff']:+.5f}  p={res['p_value']}")

    return rows, curves


def plot_corrected_vs_original(curves_by_panel, out_path):
    """Saves opening_value_betting_corrected.png: bankroll trajectories for
    both opening-side variants (honest leave-one-out, and the circular full-
    opening-panel sanity-check), log scale -- same convention as
    value_betting.plot_bankrolls."""
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
    ax.grid(True, which="both", color="#e1e0d9", linewidth=0.6)
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


def main():
    panel, P, awake, y, dates, seasons = load_full_universe()
    open_idx, close_idx = split_by_market_phase(panel)

    # ---- Part 1: does closing-side skill predict opening-side skill? ----
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

    # ---- Part 2: three variants of the BW value-betting test ----
    print(f"\n{'='*70}\nPart 2: value betting vs. {HELD_OUT_OPENING_BOOKMAKER}, three panel variants\n{'='*70}")
    match_ids = load_match_order()
    assert len(match_ids) == P.shape[0], "match_ids ordering must line up with load_full_universe()"

    bw_col = panel.index(HELD_OUT_OPENING_BOOKMAKER)
    honest_idx = [k for k in open_idx if k != bw_col]        # 12 bookmakers, BW excluded
    circular_idx = list(open_idx)                             # 13 bookmakers, BW included

    honest_rows, honest_curves = run_opening_panel_bw_test(
        panel, P, awake, y, honest_idx, "honest", match_ids)
    circular_rows, circular_curves = run_opening_panel_bw_test(
        panel, P, awake, y, circular_idx, "circular", match_ids)

    original_table = pd.read_csv(os.path.join(OUT_DIR, "value_betting_table.csv"))
    original_bw = original_table[original_table["held_out_bookmaker"] == HELD_OUT_OPENING_BOOKMAKER].copy()
    original_bw.insert(1, "panel", "leaky (25, incl. all closing)")
    original_bw.insert(3, "n_panel_bookmakers", 25)

    honest_table = pd.DataFrame(honest_rows)
    honest_table["panel"] = "honest (12, opening-only, BW excluded)"
    circular_table = pd.DataFrame(circular_rows)
    circular_table["panel"] = "circular (13, opening-only, BW INCLUDED)"

    combined = pd.concat([honest_table, circular_table, original_bw], ignore_index=True, sort=False)
    combined = combined.sort_values(["stage", "panel"])
    out_path = os.path.join(OUT_DIR, "opening_value_betting_corrected.csv")
    combined.to_csv(out_path, index=False)

    print(f"\n=== Three panel variants, vs. {HELD_OUT_OPENING_BOOKMAKER} -- saved to {out_path} ===\n")
    print(combined[["panel", "stage", "n_panel_bookmakers", "n_bets", "final_bankroll", "mean_log_growth_per_bet",
                     "p_value", "significant_95"]].to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    plot_corrected_vs_original(
        {"honest": honest_curves, "circular": circular_curves},
        os.path.join(OUT_DIR, "opening_value_betting_corrected.png"),
    )


if __name__ == "__main__":
    main()
