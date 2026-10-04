"""
Full value-betting sweep across BOTH de-vig normalizations (Basic, Shin) and
THREE OGD training-panel variants, each matched to the target bookmaker(s)
it's causally valid against:

  - "closing-trained": OGD fitted on ONLY the other closing-side bookmakers
    -- tested ONLY against closing-side targets (every closing price
    settles at roughly the same time, near kickoff, so this is valid; see
    value_betting_all_bookmakers.test_closing_bookmaker's module docstring
    for the general argument, applied here to a closing-only panel instead
    of the full one).
  - "mixed-trained": OGD fitted on the FULL 26-bookmaker panel minus the
    target -- this is value_betting.py's ORIGINAL design
    (value_betting_all_bookmakers.test_closing_bookmaker, reused directly),
    tested against ALL targets. Valid for closing-side targets; KNOWN LEAKY
    for opening-side ones (see opening_closing_quality_transfer.py) --
    included here anyway, flagged via the `leakage_risk` column, as the
    naive/uncorrected comparison point.
  - "opening-trained": OGD fitted on ONLY the other opening-side bookmakers
    -- tested against ALL targets. Always valid regardless of the target's
    phase: opening-side information is, by construction, available before
    any bookmaker's closing price forms, so using it to bet a closing
    target isn't leakage either (using older information to bet a later
    price is never a look-ahead problem, only the reverse is).

Reuses value_betting_all_bookmakers.py's test_closing_bookmaker (IS the
"mixed-trained" panel) and test_opening_bookmaker (already generalizes
correctly to closing targets too -- see that function's own logic) directly;
only "closing-trained" is new here (test_closing_trained, the mirror image
of test_opening_bookmaker using close_idx instead of open_idx). Only OGD is
tested throughout (matching value_betting.py's own scope).

Produces:
  - results/value_betting_training_panel_sweep.csv  (2 normalizations x 27
    panel-type/target combinations x 2 stages = 108 rows)
"""

import os
import numpy as np
import pandas as pd

from sleeping_experts import OUT_DIR, DATA_DIR, per_round_expert_loss, run_ogd_sleeping
from final_ranking import calibrate_full_history
from opening_vs_closing import split_by_market_phase
from value_betting import load_match_order, load_decimal_odds
from value_betting_all_bookmakers import (
    select_bookmakers, simulate_and_score, test_closing_bookmaker, test_opening_bookmaker,
)
from shin_vs_basic_comparison import load_universe_from


def test_closing_trained(panel, P, awake, y, book, close_idx, match_ids):
    """Mirror of value_betting_all_bookmakers.test_opening_bookmaker, but
    using ONLY closing-side bookmakers as the training panel -- the
    "closing-trained" variant, kept distinct from "mixed-trained"
    (test_closing_bookmaker, which uses the full panel minus the target)."""
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
    _, phat_cal, _, _ = calibrate_full_history(phat_raw, y_sub, all_true)
    return simulate_and_score(phat_raw, phat_cal, odds, y_sub, book_mask, book, "closing-trained")


def main():
    books = select_bookmakers()
    closing_books = [b for b in books if b.endswith("C")]
    print(f"Targets: {len(books)} total ({books}), {len(closing_books)} closing-side ({closing_books})\n")

    match_ids = load_match_order()  # raw odds/ordering are normalization-independent, computed once

    all_rows = []
    for norm_label, path in [("Basic", os.path.join(DATA_DIR, "odds_long.csv")),
                              ("Shin", os.path.join(DATA_DIR, "odds_long_shin.csv"))]:
        print(f"\n{'=' * 70}\n{norm_label} normalization\n{'=' * 70}")
        panel, P, awake, y, dates, seasons = load_universe_from(path)
        open_idx, close_idx = split_by_market_phase(panel)

        for book in closing_books:
            print(f"  [closing-trained] -> {book}")
            for r in test_closing_trained(panel, P, awake, y, book, close_idx, match_ids):
                r.update(normalization=norm_label, panel_type="closing-trained",
                          target_phase="closing", leakage_risk=False)
                all_rows.append(r)

        for book in books:
            print(f"  [mixed-trained]   -> {book}")
            phase = "closing" if book.endswith("C") else "opening"
            for r in test_closing_bookmaker(panel, P, awake, y, book, match_ids):
                r.update(normalization=norm_label, panel_type="mixed-trained",
                          target_phase=phase, leakage_risk=(phase == "opening"))
                all_rows.append(r)

        for book in books:
            print(f"  [opening-trained] -> {book}")
            phase = "closing" if book.endswith("C") else "opening"
            for r in test_opening_bookmaker(panel, P, awake, y, book, open_idx, match_ids):
                r.update(normalization=norm_label, panel_type="opening-trained",
                          target_phase=phase, leakage_risk=False)
                all_rows.append(r)

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


if __name__ == "__main__":
    main()
