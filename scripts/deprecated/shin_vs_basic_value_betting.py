"""
Does Shin normalization change the economics of value_betting.py's
leave-one-bookmaker-out Kelly simulation? Reruns that same design (refit the
online algorithm with the held-out bookmaker's own odds excluded from the
panel, then Kelly-bet against its real, quoted decimal odds -- see
value_betting.py's module docstring for why this avoids circularity)
independently with Basic- and Shin-normalized mixture probabilities as the
p_hat that drives the EV/Kelly-stake decision. The odds actually bet
against are the raw decimal odds either way (identical in both datasets --
de-vig method can't change what a bookmaker actually pays out), so any
difference in profit comes purely from p_hat being a different probability
estimate, not from a different betting price.

Tested algorithms: FTRL and Uniform average -- the two WORST of the four
mixture algorithms by calibrated log-loss (both under Basic and under Shin
normalization; see shin_vs_basic_table.csv), deliberately NOT OGD (the one
value_betting.py already tests, being the only algorithm
significance_test.py found to significantly beat uniform average). The
question here is different: does a better de-vig method rescue an
otherwise-weak algorithm's betting performance, not "does our best
algorithm make money".

Produces:
  - results/shin_vs_basic_value_betting_table.csv  (per algorithm x held-out
    bookmaker x normalization x raw/calibrated: bets, final bankroll/profit,
    bootstrap CI on mean log-growth per bet, + Shin-vs-Basic significance
    on log-growth per ACTIVE ROUND -- see bottom rows, kind="shin_vs_basic")
  - results/shin_vs_basic_value_betting.png  (bankroll trajectories, log
    scale, one panel per algorithm x held-out bookmaker)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import DATA_DIR, OUT_DIR, per_round_expert_loss, run_ftrl_sleeping
from final_ranking import calibrate_full_history
from significance_test import moving_block_bootstrap_test, paired_diff
from value_betting import (
    HELD_OUT_BOOKMAKERS, KELLY_MULT, STARTING_BANKROLL,
    load_match_order, load_decimal_odds, panel_excluding, simulate_kelly_betting,
)
from shin_vs_basic_comparison import load_universe_from

# the 2 worst of the 4 mixture algorithms by calibrated log-loss (see
# shin_vs_basic_table.csv) -- deliberately not OGD, which value_betting.py
# already covers
WORST_ALGORITHMS = ["FTRL", "Uniform average"]


def run_algorithm(name, P_sub, awake_sub, y):
    """Builds the mixture weights for one algorithm on one (already
    leave-one-bookmaker-out) panel -- only the two variants this script
    needs, not the full algorithm set in build_entries()."""
    N_sub = P_sub.shape[1]
    if name == "FTRL":
        loss_sub = per_round_expert_loss(P_sub, y)
        return run_ftrl_sleeping(loss_sub, awake_sub, N_sub)
    if name == "Uniform average":
        return awake_sub / awake_sub.sum(axis=1, keepdims=True)
    raise ValueError(f"unsupported algorithm: {name}")


def plot_bankrolls(curves, out_path):
    """Saves shin_vs_basic_value_betting.png: one panel per (algorithm,
    held-out bookmaker), 4 lines each (Basic/Shin x raw/calibrated) -- a
    time-series trajectory plot, same convention as value_betting.plot_bankrolls
    (log-scale bankroll, not a dot/slope chart: this is a trend over bets,
    not a one-shot ranking)."""
    panels = sorted({(algo, book) for (algo, book, _, _) in curves})
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5), facecolor="#fcfcfb")
    if n == 1:
        axes = [axes]

    style = {
        ("Basic", "raw"): dict(color="#898781", linestyle="--", linewidth=1.4),
        ("Basic", "calibrated"): dict(color="#0b0b0b", linestyle="--", linewidth=1.8),
        ("Shin", "raw"): dict(color="#eb6834", linestyle="-", linewidth=1.4),
        ("Shin", "calibrated"): dict(color="#2a78d6", linestyle="-", linewidth=1.8),
    }

    for ax, (algo, book) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        ax.axhline(1.0, color="#c3c2b7", linewidth=1.2, linestyle=":", zorder=1)
        for (a, b, norm, stage), bankroll in curves.items():
            if a != algo or b != book:
                continue
            ax.plot(np.arange(1, len(bankroll) + 1), bankroll,
                     label=f"{norm} ({stage})", zorder=2, **style[(norm, stage)])
        ax.set_yscale("log")
        ax.set_xlabel("Bet number", color="#52514e")
        ax.set_title(f"{algo}, held out: {book}", color="#0b0b0b", fontsize=11, pad=10)
        ax.grid(True, which="both", color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors="#898781")
        legend = ax.legend(frameon=False, loc="best", fontsize=8)
        for text in legend.get_texts():
            text.set_color("#0b0b0b")

    axes[0].set_ylabel("Bankroll (log scale, start = 1.0)", color="#52514e")
    fig.suptitle("Quarter-Kelly bankroll: Basic vs. Shin normalization, worst 2 algorithms",
                  color="#0b0b0b", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")


def main():
    match_ids = load_match_order()

    universes = {}
    for label, path in [("Basic", os.path.join(DATA_DIR, "odds_long.csv")),
                         ("Shin", os.path.join(DATA_DIR, "odds_long_shin.csv"))]:
        panel, P, awake, y, dates, seasons = load_universe_from(path)
        assert len(match_ids) == P.shape[0], "match_ids ordering must line up with load_universe_from()"
        universes[label] = {"panel": panel, "P": P, "awake": awake, "y": y}
    y = universes["Basic"]["y"]
    assert np.array_equal(y, universes["Shin"]["y"]), "Basic and Shin disagree on match outcomes"

    rows = []
    curves = {}
    full_log_growth = {}  # (algo, book, normalization, stage) -> full-length (per active round) log-growth

    for book in HELD_OUT_BOOKMAKERS:
        odds = load_decimal_odds(book, match_ids)  # raw decimal odds -- identical in both datasets
        book_mask = universes["Basic"]["awake"][:, universes["Basic"]["panel"].index(book)] & ~np.isnan(odds[:, 0])

        for algo in WORST_ALGORITHMS:
            print(f"\n--- {algo}, held out: {book} ---")
            for label, uni in universes.items():
                sub_panel, P_sub, awake_sub = panel_excluding(uni["panel"], uni["P"], uni["awake"], book)
                W = run_algorithm(algo, P_sub, awake_sub, y)
                phat_raw = np.einsum("tn,tnk->tk", W, P_sub)

                all_true = np.ones(len(y), dtype=bool)
                _, phat_cal, _, best_c = calibrate_full_history(phat_raw, y, all_true)

                for stage, p_hat in [("raw", phat_raw), ("calibrated", phat_cal)]:
                    growth, bet_placed = simulate_kelly_betting(p_hat, odds, y, book_mask)
                    bankroll = STARTING_BANKROLL * np.cumprod(growth[book_mask])
                    curves[(algo, book, label, stage)] = bankroll

                    log_growth_active = np.log(growth[book_mask])  # 0 on no-bet rounds
                    full_log_growth[(algo, book, label, stage)] = log_growth_active

                    log_growth_per_bet = np.log(growth[bet_placed])
                    n_bets = int(bet_placed.sum())
                    n_active = int(book_mask.sum())
                    if n_bets >= 30:
                        res = moving_block_bootstrap_test(log_growth_per_bet)
                    else:
                        res = {"mean_diff": log_growth_per_bet.mean() if n_bets else np.nan,
                               "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
                               "significant_95": False}

                    profit_pct = (bankroll[-1] - 1.0) * 100 if len(bankroll) else np.nan
                    rows.append({
                        "algorithm": algo, "held_out_bookmaker": book, "normalization": label, "stage": stage,
                        "active_rounds": n_active, "n_bets": n_bets,
                        "bet_rate_pct": 100 * n_bets / max(1, n_active),
                        "final_bankroll": bankroll[-1] if len(bankroll) else np.nan,
                        "profit_pct": profit_pct,
                        "mean_log_growth_per_bet": res["mean_diff"],
                        "ci_low": res["ci_low"], "ci_high": res["ci_high"], "p_value": res["p_value"],
                        "significant_95": res["significant_95"],
                    })
                    print(f"  [{label:5s} {stage:10s}] bets={n_bets:5d} ({100*n_bets/max(1,n_active):.1f}% of "
                          f"{n_active} active rounds)  profit={profit_pct:+.2f}%  "
                          f"mean_log_growth/bet={res['mean_diff']:+.5f}")

    table = pd.DataFrame(rows)

    # ---- Shin vs Basic significance, per algorithm x bookmaker x stage,
    # on log-growth per ACTIVE ROUND (not per bet -- the two normalizations
    # place bets on different subsets of rounds, so pairing on the full
    # active-round index, with 0 log-growth on no-bet rounds, is what makes
    # the comparison well-defined) ----
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

    plot_bankrolls(curves, os.path.join(OUT_DIR, "shin_vs_basic_value_betting.png"))


if __name__ == "__main__":
    main()
