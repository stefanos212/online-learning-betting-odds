"""
VALUE BETTING ON THE LAST FIVE SEASONS, PHASE-MATCHED.

Companion to `recent_window.py`, which covers the forecasting side. Same
window, same reason: over 10 seasons only one closing column clears any
sensible coverage floor, so the betting sweep is dominated by opening targets.
In the last five seasons B365C, BWC, VC_BVC and WHC are all available.

Design is the one `value_betting.run_training_panel_sweep` settled on:
Kelly 1/4, leave-one-bookmaker-out, calibrated probabilities, and the training
panel MATCHED to the target's market phase -- closing panel for a closing
target, opening panel for an opening one. The second is a causality
requirement: no closing price exists yet when only an opening price is posted.
The mixed (full 26) panel is also run against closing targets, as the
comparison point that shows the phase match is not cosmetic.

SIGN WARNING: this file bootstraps log-GROWTH per bet, where HIGHER is better.
That is the opposite of every log-loss table in the project.

Produces:
  - results/recent_window_value_betting.csv
"""

import os
import numpy as np
import pandas as pd

from sleeping_experts import DATA_DIR, OUT_DIR, load_match_order, load_universe_from, split_by_market_phase
from betting import (
    KELLY_MULT, fit_leave_out, score_bet, simulate_kelly_betting,
)

RECENT_SEASONS = 5
MIN_COVERAGE_PCT = 50.0   # recomputed inside the window


def main():
    match_ids = load_match_order()
    panel, P, awake, y, dates, seasons = load_universe_from(
        os.path.join(DATA_DIR, "odds_long.csv"))

    keep = sorted(set(seasons))[-RECENT_SEASONS:]
    m = np.isin(seasons, list(keep))
    P, awake, y, match_ids, seasons = P[m], awake[m], y[m], match_ids[m], seasons[m]
    T, N, _ = P.shape
    open_idx, close_idx = split_by_market_phase(panel)
    print(f"σεζόν {keep}, {T} γύροι\n")

    coverage = awake.mean(axis=0) * 100
    targets = [b for k, b in enumerate(panel)
               if coverage[k] >= MIN_COVERAGE_PCT and len(set(seasons[awake[:, k]])) > 1]
    print(f"στόχοι ({len(targets)}): {targets}\n")

    runs = []
    for book in targets:
        is_closing = book.endswith("C")
        phase = "closing" if is_closing else "opening"
        matched = close_idx if is_closing else open_idx
        runs.append((book, phase, "ταιριασμένο", matched, True))
        if is_closing:   # the comparison point, mixed panel on a closing target
            runs.append((book, phase, "πλήρες panel", list(range(N)), False))

    rows = []
    for book, phase, panel_label, idx, reduce_rounds in runs:
        print(f"  [{panel_label}] -> {book} ({phase})")
        _, phat_raw, phat_cal, odds, book_mask, y_sub, _ = fit_leave_out(
            panel, P, awake, y, idx, book, match_ids, apply_reduction=reduce_rounds)
        for stage, p_hat in (("raw", phat_raw), ("calibrated", phat_cal)):
            growth, bet_placed = simulate_kelly_betting(p_hat, odds, y_sub, book_mask)
            row, _ = score_bet(growth, bet_placed, book_mask, {
                "bookmaker": book, "phase": phase, "training_panel": panel_label,
                "stage": stage})
            rows.append(row)

    table = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "recent_window_value_betting.csv")
    table.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    cal = table[(table["stage"] == "calibrated") &
                (table["training_panel"] == "ταιριασμένο")].sort_values(
        "mean_log_growth_per_bet", ascending=False)
    print(f"\n=== Kelly {KELLY_MULT}, βαθμονομημένο, panel ταιριασμένο στη φάση ===\n")
    print(cal[["bookmaker", "phase", "n_bets", "final_bankroll",
               "mean_log_growth_per_bet", "p_value", "significant_95"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    wins = cal[cal["significant_95"] & (cal["mean_log_growth_per_bet"] > 0)]
    losses = cal[cal["significant_95"] & (cal["mean_log_growth_per_bet"] < 0)]
    print(f"\nσε {len(cal)} bookmakers: {len(wins)} σημαντικό κέρδος "
          f"({list(wins['bookmaker'])}), {len(losses)} σημαντική ζημιά "
          f"({list(losses['bookmaker'])}), {len(cal) - len(wins) - len(losses)} τίποτα")

    print("\n=== closing στόχοι: ταιριασμένο panel vs πλήρες panel ===\n")
    cmp = table[(table["stage"] == "calibrated") & (table["phase"] == "closing")]
    piv = cmp.pivot(index="bookmaker", columns="training_panel",
                    values=["mean_log_growth_per_bet", "p_value"])
    print(piv.to_string(float_format=lambda v: f"{v:.5f}"))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
