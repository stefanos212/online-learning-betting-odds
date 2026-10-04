"""
CONTEXTUAL / PER-LEAGUE SPECIALIZATION.

Kept in its OWN file (not added into sleeping_experts.py) on purpose, for
two reasons: (1) it's conceptually a different question from the rest of the
pipeline -- "does splitting the learning problem by context help?" rather
than "which algorithm is best on the pooled data?" -- and (2) so the
already-stable core files don't need to change again.

This directly tests the original thesis idea from the very first design
discussion: "eventually learning which experts are better for different
types of bets, e.g. different leagues". We have not tested that claim
anywhere else in the codebase until this file.

Method: instead of ONE OGD instance sharing weights across all 10 leagues
pooled together (what sleeping_experts.py / final_ranking.py do), run TEN
INDEPENDENT OGD instances, one per league, each starting from uniform
weights and only ever seeing that league's own chronological match sequence.
Compare each specialist's log-loss on its league's matches against the
GLOBAL (pooled) OGD's log-loss on those SAME matches -- a fair, apples-to-
apples comparison since both are scored on identical rounds, just trained
differently (shared vs. per-league weights).

Produces:
  - results/contextual_experts_table.csv   (per league: n_matches, global
                                             OGD log-loss vs. specialist
                                             log-loss, which wins, top-3
                                             bookmakers the specialist
                                             ended up trusting most)
  - results/contextual_experts.png         (per-league comparison, dot plot)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    EXCLUDED_BOOKMAKERS,
    DATA_DIR, OUT_DIR, EPS, COLORS,
    load_full_universe, load_match_order, per_round_expert_loss,
    run_ogd_sleeping, evaluate, last_awake_weight,
)
from significance_test import moving_block_bootstrap_test

# Scoped to the 5 original top-flight leagues + Scotland's top flight (6
# total), NOT all 22 divisions now in the data. Two reasons: (1) comparing
# specialists across wildly different tiers (a top flight vs. a 4th division)
# mixes leagues with very different predictability/favorite-longshot profiles
# to begin with, which would muddy the "does per-league specialization help"
# question this file exists to answer, and (2) 22 independent OGD refits is a
# lot of extra compute for a comparison this file already found (on the
# original 10 divisions) doesn't help -- no reason to expect that changes by
# adding lower divisions or non-domestic-rival leagues.
LEAGUE_NAMES = {
    "E0": "England Premier League",
    "SC0": "Scotland Premiership",
    "I1": "Italy Serie A",
    "SP1": "Spain La Liga",
    "D1": "Germany Bundesliga",
    "F1": "France Ligue 1",
}


def load_league_labels(match_ids):
    """League code for each match, aligned to the same round order load_full_universe()
    uses (load_full_universe() itself doesn't return League, so we pull it separately,
    keyed by MatchID, the same way value_betting.py pulls raw odds)."""
    long = pd.read_csv(os.path.join(DATA_DIR, "odds_long.csv"))
    meta = long.drop_duplicates("MatchID").set_index("MatchID")["League"]
    return meta.reindex(match_ids).values


def run_league_specialist(P, awake, y, league_idx):
    """Fit an independent OGD instance using ONLY this league's rounds, in
    their own chronological order (a slice of the globally-sorted tensor is
    still chronologically sorted, so no re-sorting needed)."""
    P_l, awake_l, y_l = P[league_idx], awake[league_idx], y[league_idx]
    loss_l = per_round_expert_loss(P_l, y_l)
    N = P.shape[1]
    W_l = run_ogd_sleeping(loss_l, awake_l, N)
    phat_l = np.einsum("tn,tnk->tk", W_l, P_l)
    return phat_l, W_l


def plot_league_comparison(table, out_path):
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    order = table.sort_values("specialist_log_loss")
    ypos = np.arange(len(order))[::-1]
    ax.scatter(order["global_log_loss"], ypos, color=COLORS["benchmark"], s=70,
                label="Global (pooled) OGD", zorder=3)
    ax.scatter(order["specialist_log_loss"], ypos, color=COLORS["OGD"], s=70,
                label="Per-league specialist OGD", zorder=3)
    for yp, (_, row) in zip(ypos, order.iterrows()):
        ax.plot([row["global_log_loss"], row["specialist_log_loss"]], [yp, yp],
                 color="#c3c2b7", linewidth=1.2, zorder=2)

    ax.set_yticks(ypos)
    ax.set_yticklabels(order["league"], fontsize=9, color="#0b0b0b")
    ax.set_xlabel("Log-loss on that league's matches", color="#52514e")
    ax.set_title("Global vs. per-league specialist OGD", color="#0b0b0b", fontsize=12, pad=12)
    ax.grid(True, axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", left=False)
    legend = ax.legend(frameon=False, loc="best", fontsize=8.5)
    for text in legend.get_texts():
        text.set_color("#0b0b0b")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def main():
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    match_ids = load_match_order()
    league_labels = load_league_labels(match_ids)
    assert len(league_labels) == T

    # global (pooled) OGD, exactly as in sleeping_experts.py/final_ranking.py --
    # trained ONCE sharing weights across all leagues, evaluated per-league below
    loss_global = per_round_expert_loss(P, y)
    W_global = run_ogd_sleeping(loss_global, awake, N)
    phat_global = np.einsum("tn,tnk->tk", W_global, P)
    global_logloss_per_round = -np.log(np.clip(phat_global[np.arange(T), y], EPS, 1.0))

    rows = []
    diffs_by_league = {}  # (specialist - global) per-round diffs, keyed by league, for one overall test
    for code, name in LEAGUE_NAMES.items():
        league_idx = np.where(league_labels == code)[0]
        if len(league_idx) == 0:
            continue

        phat_l, W_l = run_league_specialist(P, awake, y, league_idx)
        y_l = y[league_idx]
        specialist_ll = -np.log(np.clip(phat_l[np.arange(len(y_l)), y_l], EPS, 1.0))
        global_ll = global_logloss_per_round[league_idx]

        # which bookmakers does the specialist end up trusting most in THIS league?
        # Each bookmaker's weight the LAST time it was awake IN THIS LEAGUE, not
        # W_l[-1]: the last row is 0 for anyone that didn't quote this league's
        # final match, which would silently drop them from the ranking (and, on
        # the full panel, drops 10 of 26 bookmakers -- see
        # sleeping_experts.last_awake_weight). NaN = never awake here, so its
        # weight is just the untouched uniform init and is meaningless -> excluded.
        final_w = last_awake_weight(W_l, awake[league_idx])
        final_w = np.nan_to_num(final_w, nan=-1.0)  # never-awake -> excluded from the top-3
        top3_idx = np.argsort(final_w)[::-1][:3]
        top3 = [panel[i] for i in top3_idx]

        rows.append({
            "league": code, "league_name": name, "n_matches": len(league_idx),
            "global_log_loss": global_ll.mean(), "specialist_log_loss": specialist_ll.mean(),
            "specialist_wins": bool(specialist_ll.mean() < global_ll.mean()),
            "top3_trusted_bookmakers": ", ".join(top3),
        })
        diffs_by_league[code] = specialist_ll - global_ll
        print(f"  {code:4s} ({name:24s}) n={len(league_idx):5d}  "
              f"global={global_ll.mean():.4f}  specialist={specialist_ll.mean():.4f}  "
              f"top3={top3}")

    table = pd.DataFrame(rows).sort_values("specialist_log_loss")
    table_path = os.path.join(OUT_DIR, "contextual_experts_table.csv")
    table.to_csv(table_path, index=False)
    pd.set_option("display.width", 160)
    print(f"\n=== Per-league specialization results — saved to {table_path} ===\n")
    print(table.drop(columns=["top3_trusted_bookmakers"])
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # one overall test, pooling every league's (specialist - global) differences
    # BACK INTO THEIR ORIGINAL CHRONOLOGICAL POSITIONS before bootstrapping, so the
    # block bootstrap still resamples temporally-contiguous chunks, not a
    # league-by-league concatenation that would create artificial "seams"
    pooled_diff = np.full(T, np.nan)
    for code, diff in diffs_by_league.items():
        league_idx = np.where(league_labels == code)[0]
        pooled_diff[league_idx] = diff
    valid = ~np.isnan(pooled_diff)
    res = moving_block_bootstrap_test(pooled_diff[valid])
    print(f"\n=== Overall (specialist - global), all leagues pooled in original chronological order ===")
    print(f"mean_diff={res['mean_diff']:+.5f}  95% CI=[{res['ci_low']:+.5f}, {res['ci_high']:+.5f}]  "
          f"p={res['p_value']:.4f}  {'SIGNIFICANT' if res['significant_95'] else 'not significant'}  "
          f"(negative = specialists win overall)")

    plot_league_comparison(table, os.path.join(OUT_DIR, "contextual_experts.png"))


if __name__ == "__main__":
    main()
