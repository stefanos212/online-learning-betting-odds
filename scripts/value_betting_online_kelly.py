"""
Kelly betting where the STAKE FRACTION ITSELF is learned ONLINE via
projected OGD on the negative-log-wealth loss, instead of being computed
each round from the classical closed-form Kelly formula
f* = (pO - 1)/(O - 1) plugged in from a point-estimate probability
(value_betting.kelly_fraction). This is a THIRD Online Convex Optimization
stage on top of the pipeline (mixture weights -> optional calibration
temperature -> now stake size), following the exact same projected-OGD
recipe as sleeping_experts.run_ogd_sleeping() and
calibration_correction.py's temperature_calibrate() -- the "Online
Portfolio Selection" framing of Kelly betting (Cover, 1991), applied here
to the single highest-EV outcome per round (same outcome-selection rule as
value_betting.simulate_kelly_betting) rather than a full multi-asset
portfolio.

THE ONLINE STAKE LEARNER
-------------------------
On a round where a bet is placed (positive EV, same selection as the
classical version), with odds O for the chosen outcome and win indicator w
(revealed AFTER the stake f is played), the realized loss is
    loss(f) = -log(1 - f + f*O*w)      (negative log-wealth growth)
which is convex in f (composition of -log with an affine, increasing,
positive function). Its gradient at the played f is
    win (w=1):  d loss/df = -(O-1) / (1 - f + f*O)   [>= 1, so this stays finite]
    lose (w=0): d loss/df =  1 / (1 - f)
updated via projected OGD, f_{t+1} = clip(f_t - eta_t * grad_t, 0, F_MAX),
eta_t = D / (M * sqrt(t+1)) with D = F_MAX (diameter of the feasible
interval) and M = max(1/(1-F_MAX), max_odds - 1) (a genuine bound on the
per-round sub-gradient magnitude given the F_MAX cap -- not an arbitrary
constant). F_MAX = 0.5: never stake more than half the bankroll on one bet
-- this is what keeps the lose-case gradient finite (no epsilon hacks
needed), and is the online-learned analogue of the project's existing
"quarter-Kelly" risk cap on the classical side.

f's own round-counter t starts at 0 on the FIRST round a bet is actually
placed (synced to whenever the closing-trained mixture itself starts
producing usable, positive-EV signals), not from round 0 of the whole
76,584-match history, most of which never triggers a bet.

Runs the SAME "closing-on-closing" comparison as
value_betting_closing_trained.py (OGD mixture trained on ONLY the other
closing-side bookmakers, bet against each closing-side target -- causally
valid, no look-ahead risk), for both de-vig normalizations (Basic, Shin),
comparing this online-learned stake against the classical fixed-formula
Kelly on the exact same bets (same p_hat, same outcome selected each round
-- only the stake SIZE differs), so a direct paired bootstrap test is valid.

Produces:
  - results/value_betting_online_kelly_table.csv
  - results/value_betting_online_kelly_significance.csv  (online vs.
    classical, paired per bet)
  - results/value_betting_online_kelly.png  (bankroll trajectories: online
    stake vs. classical Kelly, one panel per closing-side target)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import OUT_DIR, DATA_DIR, per_round_expert_loss, run_ogd_sleeping, load_match_order, load_universe_from, split_by_market_phase
from final_ranking import calibrate_full_history
from significance_test import moving_block_bootstrap_test
from betting import (
    STARTING_BANKROLL, load_decimal_odds, select_bookmakers, simulate_kelly_betting,
)

F_MAX = 0.5  # online-learned stake never exceeds this -- also bounds the lose-case gradient


def simulate_online_kelly(p_hat, odds, y, mask, F_MAX=F_MAX):
    """Same round-by-round outcome selection as
    value_betting.simulate_kelly_betting (bet the single highest-positive-EV
    outcome, if any), but the stake fraction f is learned ONLINE via
    projected OGD on the realized negative-log-wealth loss, instead of the
    closed-form Kelly formula. See module docstring for the derivation."""
    T = len(y)
    growth = np.ones(T)
    bet_placed = np.zeros(T, dtype=bool)
    f_trace = np.full(T, np.nan)

    valid_odds = odds[mask][np.isfinite(odds[mask])]
    max_odds = float(np.nanmax(valid_odds)) if valid_odds.size else 2.0
    D = F_MAX
    M = max(1.0 / (1.0 - F_MAX), max_odds - 1.0)

    f = 0.0
    t_bet = 0
    for t in range(T):
        if not mask[t]:
            continue
        ev = p_hat[t] * odds[t] - 1.0
        k = int(np.argmax(ev))
        if ev[k] <= 0:
            continue

        bet_placed[t] = True
        f_trace[t] = f
        O = odds[t, k]
        win = (y[t] == k)
        growth[t] = (1.0 - f + f * O) if win else (1.0 - f)

        grad = -(O - 1.0) / growth[t] if win else 1.0 / growth[t]
        eta = D / (M * np.sqrt(t_bet + 1))
        f = float(np.clip(f - eta * grad, 0.0, F_MAX))
        t_bet += 1

    return growth, bet_placed, f_trace


def fit_closing_on_closing(panel, P, awake, y, book, close_idx, match_ids):
    """Same OGD fit as value_betting_closing_trained.run_closing_on_closing
    (closing-only training panel, target excluded), but returns the raw
    ingredients (phat_raw, phat_cal, odds, mask, y_sub) instead of already
    running the classical Kelly simulation, so both the classical and the
    online-learned stake mechanisms can be applied to the identical bets."""
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
    return phat_raw, phat_cal, odds, book_mask, y_sub


def score(growth, bet_placed, book_mask, label, book, stage):
    bankroll = STARTING_BANKROLL * np.cumprod(growth[book_mask])
    log_growth_per_bet = np.log(growth[bet_placed])
    n_bets = int(bet_placed.sum())
    if n_bets >= 30:
        res = moving_block_bootstrap_test(log_growth_per_bet)
    else:
        res = {"mean_diff": log_growth_per_bet.mean() if n_bets else np.nan,
               "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "significant_95": False}
    row = {
        "bookmaker": book, "stake_method": label, "stage": stage, "n_bets": n_bets,
        "final_bankroll": bankroll[-1] if len(bankroll) else np.nan,
        "mean_log_growth_per_bet": res["mean_diff"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
        "p_value": res["p_value"], "significant_95": res["significant_95"],
    }
    return row, bankroll


def plot_comparison(curves_by_target, targets, out_path):
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5), facecolor="#fcfcfb")
    if n == 1:
        axes = [axes]
    style = {
        ("Basic", "classical"): dict(color="#898781", linestyle="--", linewidth=1.4),
        ("Basic", "online"): dict(color="#0b0b0b", linestyle="--", linewidth=1.8),
        ("Shin", "classical"): dict(color="#eb6834", linestyle="-", linewidth=1.4),
        ("Shin", "online"): dict(color="#2a78d6", linestyle="-", linewidth=1.8),
    }
    for ax, target in zip(axes, targets):
        ax.set_facecolor("#fcfcfb")
        ax.axhline(1.0, color="#c3c2b7", linewidth=1.2, linestyle=":", zorder=1)
        for (norm, method), bankroll in curves_by_target[target].items():
            ax.plot(np.arange(1, len(bankroll) + 1), bankroll,
                     label=f"{norm} ({method})", zorder=2, **style[(norm, method)])
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
    fig.suptitle("Closing-on-closing: online-learned stake (OCO) vs. classical Kelly formula, calibrated p_hat",
                  color="#0b0b0b", fontsize=12.5, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")


def main():
    books = select_bookmakers()
    targets = [b for b in books if b.endswith("C")]
    print(f"Closing-side targets: {targets}\n")

    match_ids = load_match_order()

    rows, sig_rows = [], []
    curves_by_target = {t: {} for t in targets}

    for norm_label, path in [("Basic", os.path.join(DATA_DIR, "odds_long.csv")),
                              ("Shin", os.path.join(DATA_DIR, "odds_long_shin.csv"))]:
        print(f"\n{'=' * 60}\n{norm_label} normalization\n{'=' * 60}")
        panel, P, awake, y, dates, seasons = load_universe_from(path)
        _, close_idx = split_by_market_phase(panel)

        for book in targets:
            phat_raw, phat_cal, odds, book_mask, y_sub = fit_closing_on_closing(
                panel, P, awake, y, book, close_idx, match_ids
            )
            for stage, p_hat in [("raw", phat_raw), ("calibrated", phat_cal)]:
                g_classical, bp_classical = simulate_kelly_betting(p_hat, odds, y_sub, book_mask)
                g_online, bp_online, f_trace = simulate_online_kelly(p_hat, odds, y_sub, book_mask)
                assert np.array_equal(bp_classical, bp_online), "the two methods must agree on WHICH rounds to bet"

                row_c, bank_c = score(g_classical, bp_classical, book_mask, "classical", book, stage)
                row_o, bank_o = score(g_online, bp_online, book_mask, "online", book, stage)
                rows.extend([row_c, row_o])
                if stage == "calibrated":
                    curves_by_target[book][(norm_label, "classical")] = bank_c
                    curves_by_target[book][(norm_label, "online")] = bank_o

                diff, n = np.log(g_online[bp_online]) - np.log(g_classical[bp_classical]), int(bp_online.sum())
                res = moving_block_bootstrap_test(diff)
                sig_rows.append({"normalization": norm_label, "bookmaker": book, "stage": stage,
                                  "mean_final_f": float(np.nanmean(f_trace)), **res})

                print(f"  {book:8s} {stage:10s}  classical final={row_c['final_bankroll']:.4f}  "
                      f"online final={row_o['final_bankroll']:.4f}  mean_f={np.nanmean(f_trace):.4f}  "
                      f"online-vs-classical p={res['p_value']}")

    table = pd.DataFrame(rows)
    table_path = os.path.join(OUT_DIR, "value_betting_online_kelly_table.csv")
    table.to_csv(table_path, index=False)

    sig_table = pd.DataFrame(sig_rows)
    sig_path = os.path.join(OUT_DIR, "value_betting_online_kelly_significance.csv")
    sig_table.to_csv(sig_path, index=False)

    pd.set_option("display.width", 160)
    print(f"\n=== Online-learned vs. classical Kelly -- saved to {table_path} ===\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    # SIGN WARNING: unlike every bootstrap row elsewhere in the project, the
    # quantity here is log-GROWTH (wealth), not a loss -- diff = log-growth
    # (online) - log-growth (classical), so POSITIVE = online better. Do not
    # copy significance_test.py's "negative = first-named is better" reading
    # onto this table (see CLAUDE.md's sign-convention gotcha).
    print(f"\n=== Online vs. classical, paired bootstrap on log-growth per bet "
          f"(negative = online WORSE, i.e. lower wealth growth) -- saved to {sig_path} ===\n")
    print(sig_table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    plot_comparison(curves_by_target, targets, os.path.join(OUT_DIR, "value_betting_online_kelly.png"))


if __name__ == "__main__":
    main()
