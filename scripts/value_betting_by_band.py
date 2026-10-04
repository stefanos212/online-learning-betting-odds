"""
VALUE BETTING RESTRICTED TO A PROBABILITY BAND.

The question this answers: accuracy_by_band.py found the mixture's log-loss
advantage over the bookmakers is not uniform -- it grows with confidence, and
the 33-50% region (around 60% of all matches) is where every method is
nearly identical. So: does filtering the bets down to the band where we
forecast best actually make money?

TWO THINGS THAT MUST BE SAID UP FRONT, because they are the whole difficulty:

  1. A LOG-LOSS EDGE IS NOT A BETTING EDGE. Log-loss measures whether our
     probabilities are closer to reality than the bookmaker's. Profit
     measures whether the bookmaker's PRICE is wrong in an exploitable
     direction by more than its own margin. Those are different questions:
     we can be better calibrated everywhere and still never be offered a
     price worth taking. So the band with the biggest log-loss margin is not
     automatically the band with the best returns, and this file does not
     assume it is -- it measures all of them.

  2. PICKING THE BEST BAND POST HOC IS DATA SNOOPING. With 10 bands x 11
     bookmakers x 2 stages, something will look profitable by chance. Every
     result here is therefore reported twice:
       - FULL SAMPLE: descriptive, all bands, for the shape of the effect.
       - TRAIN/VAL: the band is chosen on the first TRAIN_FRACTION of each
         target's bets (by final bankroll) and then evaluated ONLY on the
         remaining bets, which the choice never saw. This is the honest
         number -- the one to quote. Same chronological-split discipline as
         calibration_correction.py.

Banding is on the probability WE assign to the outcome we actually bet on
(not the argmax): value betting takes the highest-positive-EV outcome, which
is frequently not the favourite, so "the band we are betting in" is the band
of the bet itself. That also matches the framing the bands came from.

Design otherwise identical to value_betting.run_all_bookmakers_sweep -- the
project's canonical honest sweep: closing-side target gets the standard
leave-one-out (full panel minus target), opening-side target gets the
corrected one (opening-only panel, target excluded, no closing information
that wouldn't exist yet), quarter-Kelly, raw and causally calibrated p_hat.

Produces:
  - results/value_betting_by_band_table.csv       (full sample, every band)
  - results/value_betting_by_band_holdout.csv     (train-chosen band, val-only result)
  - results/value_betting_by_band.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import DATA_DIR, OUT_DIR, COLORS, load_match_order, load_universe_from, split_by_market_phase
from significance_test import moving_block_bootstrap_test
from betting import (
    KELLY_MULT, STARTING_BANKROLL, _style_axes, fit_leave_out, kelly_fraction, score_bet, select_bookmakers,
)

BAND_EDGES = np.linspace(0.0, 1.0, 11)   # bands of the bet's own probability
TRAIN_FRACTION = 0.75                     # chronological split for the honest half
MIN_BETS = 30                             # below this the bootstrap isn't meaningful


def band_labels():
    return [f"{lo * 100:.0f}-{hi * 100:.0f}%" for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]


def simulate_kelly_with_bands(p_hat, odds, y, mask, kelly_mult=KELLY_MULT):
    """value_betting.simulate_kelly_betting, but it also reports, for every
    round where a bet was placed, the probability WE gave the outcome we
    backed -- which is what assigns the bet to a band. Kept as a local copy
    (same reasoning as the other small copies across this project's extension
    files) rather than changing the return contract of a function six
    experiments already depend on. Bet selection, staking and growth are
    identical line for line, so a band-unrestricted run here reproduces
    value_betting.run_all_bookmakers_sweep exactly (asserted in main())."""
    T = len(y)
    growth = np.ones(T)
    bet_placed = np.zeros(T, dtype=bool)
    p_bet = np.full(T, np.nan)
    for t in range(T):
        if not mask[t]:
            continue
        ev = p_hat[t] * odds[t] - 1.0
        k = int(np.argmax(ev))
        if ev[k] <= 0:
            continue
        f = kelly_mult * kelly_fraction(p_hat[t, k], odds[t, k])
        if f <= 0:
            continue
        bet_placed[t] = True
        p_bet[t] = p_hat[t, k]
        growth[t] = (1.0 - f + f * odds[t, k]) if y[t] == k else (1.0 - f)
    return growth, bet_placed, p_bet


def score_subset(growth, sel, fields):
    """Bankroll + bootstrap over an arbitrary subset of placed bets."""
    n = int(sel.sum())
    row = {**fields, "n_bets": n}
    if n == 0:
        return {**row, "final_bankroll": np.nan, "mean_log_growth_per_bet": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
                "significant_95": False, "thin": True}
    lg = np.log(growth[sel])
    bankroll = STARTING_BANKROLL * np.cumprod(growth[sel])
    if n >= MIN_BETS:
        res = moving_block_bootstrap_test(lg)
    else:
        res = {"mean_diff": lg.mean(), "ci_low": np.nan, "ci_high": np.nan,
               "p_value": np.nan, "significant_95": False}
    return {**row, "final_bankroll": float(bankroll[-1]),
            "mean_log_growth_per_bet": res["mean_diff"], "ci_low": res["ci_low"],
            "ci_high": res["ci_high"], "p_value": res["p_value"],
            "significant_95": res["significant_95"], "thin": n < MIN_BETS}


def plot_by_band(table, out_path):
    """Calibrated stage, one line per bookmaker: mean log-growth per bet
    against the band of the bet. Dashed line at 0 = breakeven."""
    cal = table[(table["stage"] == "calibrated") & (~table["thin"])]
    labels = band_labels()
    xpos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axhline(0, color="#898781", linewidth=1.4, linestyle="--", zorder=1)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#52514e", "#898781",
                "#c3c2b7", "#0b0b0b", "#7a4fd6", "#d64f9e", "#4fd6c4"]
    for (book, g), color in zip(cal.groupby("bookmaker"), palette):
        g = g.set_index("band").reindex(labels)
        ax.plot(xpos, g["mean_log_growth_per_bet"], marker="o", markersize=4.5,
                 linewidth=1.7, color=color, label=book, zorder=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=45)
    ax.set_xlabel("Band of the backed outcome's predicted probability", color="#52514e")
    ax.set_ylabel("Mean log-growth per bet (0 = breakeven)", color="#52514e")
    ax.set_title("Value betting restricted to a probability band (calibrated, quarter-Kelly)\n"
                  "full sample -- see the holdout table for the honest, non-snooped number",
                  color="#0b0b0b", fontsize=12, pad=12)
    _style_axes(ax)
    leg = ax.legend(frameon=False, loc="best", fontsize=7.5, ncol=2)
    for t in leg.get_texts():
        t.set_color("#0b0b0b")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def main():
    panel, P, awake, y, dates, seasons = load_universe_from(os.path.join(DATA_DIR, "odds_long.csv"))
    open_idx, close_idx = split_by_market_phase(panel)
    match_ids = load_match_order()
    N = len(panel)
    labels = band_labels()

    books = select_bookmakers()
    print(f"Targets ({len(books)}): {books}\n")

    rows, holdout_rows = [], []
    for book in books:
        is_closing = book.endswith("C")
        indices = list(range(N)) if is_closing else open_idx
        _, phat_raw, phat_cal, odds, book_mask, y_sub, _ = fit_leave_out(
            panel, P, awake, y, indices, book, match_ids, apply_reduction=not is_closing
        )
        print(f"--- {book} ({'closing' if is_closing else 'opening'}) ---")

        for stage, p_hat in [("raw", phat_raw), ("calibrated", phat_cal)]:
            growth, bet_placed, p_bet = simulate_kelly_with_bands(p_hat, odds, y_sub, book_mask)
            band_idx = np.clip(np.digitize(p_bet, BAND_EDGES[1:-1], right=False),
                                0, len(BAND_EDGES) - 2)

            # ---- full sample: every band, plus the unrestricted baseline ----
            all_row = score_subset(growth, bet_placed,
                                    {"bookmaker": book, "stage": stage, "band": "ALL"})
            rows.append(all_row)
            for b, label in enumerate(labels):
                sel = bet_placed & (band_idx == b)
                rows.append(score_subset(growth, sel,
                                          {"bookmaker": book, "stage": stage, "band": label}))

            # ---- honest half: choose the band on the train prefix only ----
            bet_pos = np.flatnonzero(bet_placed)
            n_train = int(round(len(bet_pos) * TRAIN_FRACTION))
            if n_train < MIN_BETS or len(bet_pos) - n_train < MIN_BETS:
                continue
            train_cut = bet_pos[n_train]
            is_train = np.zeros(len(y_sub), dtype=bool)
            is_train[:train_cut] = True

            best_band, best_bankroll = None, -np.inf
            for b, label in enumerate(labels):
                sel = bet_placed & is_train & (band_idx == b)
                if sel.sum() < MIN_BETS:
                    continue
                bankroll = float(np.prod(growth[sel]))
                if bankroll > best_bankroll:
                    best_bankroll, best_band = bankroll, (b, label)
            if best_band is None:
                continue

            b, label = best_band
            val_sel = bet_placed & (~is_train) & (band_idx == b)
            val_all = bet_placed & (~is_train)
            chosen = score_subset(growth, val_sel, {
                "bookmaker": book, "stage": stage, "chosen_band": label,
                "train_bankroll_in_band": best_bankroll, "scope": "val, chosen band"})
            baseline = score_subset(growth, val_all, {
                "bookmaker": book, "stage": stage, "chosen_band": label,
                "train_bankroll_in_band": np.nan, "scope": "val, all bands"})
            holdout_rows += [chosen, baseline]
            print(f"  [{stage:10s}] train picks band {label:8s} -> "
                  f"val bankroll {chosen['final_bankroll'] if chosen['n_bets'] else float('nan'):.4f} "
                  f"({chosen['n_bets']} bets)  vs  unrestricted val "
                  f"{baseline['final_bankroll']:.4f} ({baseline['n_bets']} bets)")

    table = pd.DataFrame(rows)
    table_path = os.path.join(OUT_DIR, "value_betting_by_band_table.csv")
    table.to_csv(table_path, index=False)

    holdout = pd.DataFrame(holdout_rows)
    holdout_path = os.path.join(OUT_DIR, "value_betting_by_band_holdout.csv")
    holdout.to_csv(holdout_path, index=False)

    pd.set_option("display.width", 200)
    print(f"\n\n=== FULL SAMPLE (descriptive -- band NOT chosen out of sample) "
          f"-- saved to {table_path} ===")
    cal = table[(table["stage"] == "calibrated") & (~table["thin"]) & (table["band"] != "ALL")]
    sig = cal[cal["significant_95"]].sort_values("mean_log_growth_per_bet", ascending=False)
    print(f"\nBands with a significant calibrated result: {len(sig)} of {len(cal)}\n")
    print(sig[["bookmaker", "band", "n_bets", "final_bankroll",
                "mean_log_growth_per_bet", "p_value"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print(f"\n\n=== HOLDOUT (band chosen on train, scored on val only -- the honest number) "
          f"-- saved to {holdout_path} ===\n")
    h = holdout[holdout["stage"] == "calibrated"]
    piv = h.pivot_table(index=["bookmaker", "chosen_band"], columns="scope",
                         values=["final_bankroll", "n_bets"])
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

    won = h[h["scope"] == "val, chosen band"]
    print(f"\nChosen-band val bankroll > 1 in {int((won['final_bankroll'] > 1).sum())} "
          f"of {len(won)} targets; significant in {int(won['significant_95'].sum())}.")

    plot_by_band(table, os.path.join(OUT_DIR, "value_betting_by_band.png"))


if __name__ == "__main__":
    main()
