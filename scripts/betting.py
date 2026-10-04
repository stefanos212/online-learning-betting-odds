"""
THE BETTING PRIMITIVES, SHARED BY EVERY SIMULATION IN THE PROJECT.

These used to live inside value_betting.py, which meant six other scripts
imported them from a sibling *experiment* rather than from a library: thirty
import sites reaching into a file whose job is to run a particular set of runs
and write a particular set of CSVs. Pulling them out leaves value_betting.py as
the experiment it is meant to be, and gives the satellites something stable to
depend on.

What belongs here is anything a second simulation would need: how to size a bet,
how to run one, how to fit the mixture it bets with, and how to score the result.
What stays in value_betting.py is the choice of which runs to perform and how to
present them.

SIGN WARNING, since it bites repeatedly: score_bet() bootstraps log-GROWTH per
bet, where HIGHER is better. That is the opposite convention to every log-loss
table in the project, where lower wins.
"""

import os

import numpy as np
import pandas as pd

from sleeping_experts import (
    DATA_DIR, OUT_DIR, per_round_expert_loss, run_ogd_sleeping,
)
from final_ranking import calibrate_full_history
from significance_test import moving_block_bootstrap_test

KELLY_MULT = 0.25   # "quarter Kelly" -- a fractional stake is standard practice to reduce the
                    # risk of large drawdowns from probability-estimate error
STARTING_BANKROLL = 1.0
MIN_COVERAGE_PCT = 50.0  # default floor for target selection in the sweeps


def load_decimal_odds(book, match_ids):
    """Raw decimal odds (not de-vig'd probabilities) quoted by `book`,
    aligned to `match_ids` order. NaN where that bookmaker didn't quote
    that match. Always read from the Basic odds_long.csv -- the raw odds
    columns are identical in odds_long_shin.csv (de-vig method can't change
    what a bookmaker actually pays out), so there's no need to pick a file
    per normalization here."""
    long = pd.read_csv(os.path.join(DATA_DIR, "odds_long.csv"))
    sub = long[long["Bookmaker"] == book].set_index("MatchID")[["odds_H", "odds_D", "odds_A"]]
    sub = sub.reindex(match_ids)
    return sub[["odds_H", "odds_D", "odds_A"]].values  # (T, 3)


def select_bookmakers(min_coverage=MIN_COVERAGE_PCT):
    """Every bookmaker with coverage_pct >= min_coverage that isn't flagged
    single_season (see results_table.csv / CLAUDE.md gotcha) -- thin/
    single-season entries would give too few bets for a meaningful
    bootstrap test and their apparent performance is confounded with one
    season's difficulty anyway."""
    results = pd.read_csv(os.path.join(OUT_DIR, "results_table.csv"))
    books = results[(results["type"] == "bookmaker") &
                     (results["coverage_pct"] >= min_coverage) &
                     (~results["single_season"])]
    return books.sort_values("coverage_pct", ascending=False)["name"].tolist()


def kelly_fraction(p, odds):
    """Full Kelly stake fraction for a single bet: f* = p - (1-p)/(odds-1).
    Clipped to [0, 1] -- we never bet a negative or leveraged stake."""
    b = odds - 1.0
    f = p - (1.0 - p) / b
    return np.clip(f, 0.0, 1.0)


def simulate_kelly_betting(p_hat, odds, y, mask, kelly_mult=KELLY_MULT):
    """
    Bet on the single outcome (H/D/A) with the highest positive expected
    value each round (if any), sized at `kelly_mult` times the full Kelly
    fraction. Rounds where the bookmaker didn't quote odds (mask False), or
    where no outcome has positive EV, are "no bet" (growth = 1.0, bankroll
    unchanged that round).

    Returns:
      growth[t]      -- bankroll multiplier for round t (1.0 if no bet)
      bet_placed[t]  -- whether a bet was actually placed that round
    """
    T = len(y)
    growth = np.ones(T)
    bet_placed = np.zeros(T, dtype=bool)
    for t in range(T):
        if not mask[t]:
            continue
        ev = p_hat[t] * odds[t] - 1.0          # expected value per unit staked, one per outcome
        k = int(np.argmax(ev))
        if ev[k] <= 0:
            continue                            # no positive-EV outcome this round -> skip
        f = kelly_mult * kelly_fraction(p_hat[t, k], odds[t, k])
        if f <= 0:
            continue
        bet_placed[t] = True
        if y[t] == k:
            growth[t] = 1.0 - f + f * odds[t, k]  # stake returned as (odds * stake) on a win
        else:
            growth[t] = 1.0 - f                    # stake lost
    return growth, bet_placed


def fit_leave_out(panel, P, awake, y, indices, target_book, match_ids, exclude_target=True,
                   apply_reduction=True):
    """THE fitting routine behind every betting experiment: OGD fitted on
    `indices` (a column-index subset -- the full panel, or just open_idx, or
    just close_idx), optionally excluding `target_book` from within them,
    then causally calibrated the same way as final_ranking.py.

    Replaces what used to be five near-identical "leave-one-out fit"
    functions copy-pasted across value_betting_all_bookmakers.py /
    value_betting_closing_trained.py / value_betting_training_panel_sweep.py
    / opening_closing_quality_transfer.py -- they differed only in which
    indices were passed, whether the target itself stayed inside them
    (exclude_target=False is used exactly once, for the "circular" sanity
    check: betting against a bookmaker using a mixture that still includes
    its own price), and one more axis:

    `apply_reduction` -- when `indices` is a SMALL subset (open_idx or
    close_idx, ~13 bookmakers), that subset does NOT inherit
    load_full_universe()'s ">=1 of 26 awake every round" guarantee, so
    rounds must be further restricted to those where >=1 of the kept
    columns is awake AND target_book itself has usable odds. But for the
    FULL-panel-minus-target ("mixed-trained") case the original code never
    applied this reduction at all -- it dropped one column and ran OGD
    across every round unrestricted. Passing apply_reduction=False
    reproduces that exactly; every caller using the full panel
    (list(range(N))) must pass it.

    Returns (sub_panel, phat_raw, phat_cal, odds, book_mask, y_sub, best_c).
    `odds` and `book_mask` are already aligned to y_sub, ready to pass
    straight into simulate_kelly_betting."""
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
    W_ogd = run_ogd_sleeping(loss_sub, awake_sub, N_sub)
    phat_raw = np.einsum("tn,tnk->tk", W_ogd, P_sub)

    all_true = np.ones(len(y_sub), dtype=bool)
    _, phat_cal, _, best_c = calibrate_full_history(phat_raw, y_sub, all_true)
    return sub_panel, phat_raw, phat_cal, odds, book_mask, y_sub, best_c


def score_bet(growth, bet_placed, book_mask, fields):
    """Shared Kelly-simulation scoring: final bankroll + bootstrap test on
    mean log-growth per bet, packaged as one result row. `fields` is merged
    in first so each caller can label the row (bookmaker, stage, etc.)."""
    bankroll = STARTING_BANKROLL * np.cumprod(growth[book_mask])
    log_growth_per_bet = np.log(growth[bet_placed])
    n_bets = int(bet_placed.sum())
    n_active = int(book_mask.sum())
    if n_bets >= 30:  # bootstrap needs a reasonable sample to be meaningful
        res = moving_block_bootstrap_test(log_growth_per_bet)
    else:
        res = {"mean_diff": log_growth_per_bet.mean() if n_bets else np.nan,
               "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "significant_95": False}
    row = {
        **fields, "active_rounds": n_active, "n_bets": n_bets,
        "bet_rate_pct": 100 * n_bets / max(1, n_active),
        "final_bankroll": bankroll[-1] if len(bankroll) else np.nan,
        "mean_log_growth_per_bet": res["mean_diff"],
        "ci_low": res["ci_low"], "ci_high": res["ci_high"],
        "p_value": res["p_value"], "significant_95": res["significant_95"],
    }
    return row, bankroll


def _style_axes(ax):
    """The project's plot styling, as used by the betting figures."""
    ax.grid(True, which="both", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")
