"""
CAN WE JUST USE THE SHARP BOOKMAKER AS THE FORECAST?

This is the standard professional recipe, and the natural challenge to the
whole thesis: Pinnacle's closing price (PSC) is the best single forecaster in
the panel (final_ranking_table.csv), so instead of learning a mixture, treat
PSC's de-vigged probabilities as truth and bet them against the softer
bookmakers. If that makes more money than the OCO mixture, the mixture is
economically redundant however good its log-loss is.

TIMING CONSTRAINT -- why only closing targets. PSC is a CLOSING price, set
just before kickoff. Using it to bet against an OPENING price is exactly the
look-ahead bias this project already found and fixed once (see
value_betting.run_bw_leakage_check): the closing line does not exist when the
opening line is posted. So the sharp anchor is only admissible against other
CLOSING bookmakers, and this file restricts itself to those. That is also
why the target list here is not `select_bookmakers()`'s 11 -- it is the
closing side only, with the coverage floor dropped to 15% so IWC and BFEC
are included (single-season entries stay excluded).

FOUR FORECASTERS, identical bets otherwise (quarter-Kelly, highest
positive-EV outcome, same rounds):
  1. PSC raw            -- the sharp price, de-vigged, used as-is
  2. PSC calibrated     -- same, after the project's causal temperature scaling
  3. OGD raw            -- the mixture, target excluded (canonical design)
  4. OGD calibrated     -- the headline configuration of the whole project

NOTE the mixture already CONTAINS PSC: for a closing target the panel is the
full 26 minus the target, PSC included. So this is not "sharp vs learned" in
a clean sense -- it is "PSC alone vs a blend that includes PSC". The
question is whether blending PSC with 24 lesser experts helps or dilutes.

Every comparison is restricted to rounds where PSC, the target and the odds
are ALL available, so all four forecasters bet on the same opportunity set.
PSC vs OGD is then paired per active round on log-growth (they place
different bets, so per-bet pairing is not available -- same device as
value_betting_ogd_vs_ftrl.py).

SIGN: the paired table bootstraps log-GROWTH, so POSITIVE mean_diff means
the first-named forecaster is BETTER (see CLAUDE.md's sign gotcha).

Produces:
  - results/value_betting_sharp_anchor_table.csv
  - results/value_betting_sharp_anchor_significance.csv
  - results/value_betting_sharp_anchor.png
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
from final_ranking import calibrate_full_history, COVERAGE_THRESHOLD
from significance_test import moving_block_bootstrap_test
from betting import (
    _style_axes, load_decimal_odds, score_bet, simulate_kelly_betting,
)

# (anchor, market phase). The anchor is the best Tier-A bookmaker of its own
# phase by log-loss: PSC for closing, VC_BV for opening (LB scores better on
# the opening side but covers only 27% of matches over 3 seasons, so its
# apparent quality is not a trustworthy basis for an anchor -- see CLAUDE.md
# on the coverage threshold).
ANCHORS = [("PSC", "closing"), ("VC_BV", "opening")]
MIN_TARGET_COVERAGE = 15.0   # lower than value_betting's 50% so IWC/BFEC qualify
FORECASTER_COLORS = {"anchor raw": "#c3c2b7", "anchor calibrated": "#52514e",
                     "OGD raw": "#f0a883", "OGD calibrated": COLORS["OGD"]}


def phase_targets(panel, awake, seasons, anchor, phase):
    """Bookmakers of the anchor's own market phase, other than the anchor,
    with enough coverage and more than one season (single-season entries are
    confounded with that season's difficulty -- see CLAUDE.md)."""
    open_idx, close_idx = split_by_market_phase(panel)
    idx = close_idx if phase == "closing" else open_idx
    out = []
    for k in idx:
        name = panel[k]
        if name == anchor:
            continue
        cov = awake[:, k].mean() * 100
        n_seasons = len(set(seasons[awake[:, k]]))
        if cov >= MIN_TARGET_COVERAGE and n_seasons > 1:
            out.append((name, cov, n_seasons))
    return sorted(out, key=lambda r: -r[1])


def plot_anchor(table, out_path):
    """One row per target, one dot per forecaster: mean log-growth per bet.
    Dashed line at breakeven."""
    cal = table[~table["thin"]] if "thin" in table else table
    cal = cal.assign(label=cal["bookmaker"] + "  [" + cal["anchor"] + "]")
    order = (cal[cal["forecaster"] == "OGD calibrated"]
             .sort_values("mean_log_growth_per_bet")["label"].tolist())
    fig, ax = plt.subplots(figsize=(9.5, 0.6 * len(order) + 2.4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axvline(0, color="#898781", linewidth=1.3, linestyle="--", zorder=1)

    for i, book in enumerate(order):
        sub = cal[cal["label"] == book]
        for name, color in FORECASTER_COLORS.items():
            r = sub[sub["forecaster"] == name]
            if r.empty:
                continue
            ax.scatter([r["mean_log_growth_per_bet"].iloc[0]], [i], s=75, color=color,
                        zorder=3, edgecolor="#fcfcfb", linewidth=1.0)
        vals = sub["mean_log_growth_per_bet"]
        ax.plot([vals.min(), vals.max()], [i, i], color="#e1e0d9", linewidth=1.3, zorder=2)

    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=9, color="#0b0b0b")
    ax.set_xlabel("Mean log-growth per bet (0 = breakeven, quarter-Kelly)", color="#52514e")
    ax.set_title("Betting the best single bookmaker of the phase vs. the learned mixture\n"
                  "[PSC] = closing anchor, [VC_BV] = opening anchor; same rounds, same staking",
                  color="#0b0b0b", fontsize=11.5, pad=12)
    ax.grid(True, axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", left=False)
    legend = ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="none", color=c, markersize=8, label=n)
        for n, c in FORECASTER_COLORS.items()],
        frameon=False, loc="best", fontsize=8.5)
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

    rows, sig_rows = [], []
    for anchor, phase in ANCHORS:
        anchor_col = panel.index(anchor)
        # the mixture is restricted to the anchor's own phase for an opening
        # anchor (closing prices don't exist yet at opening-bet time); a
        # closing anchor can use the whole panel
        panel_idx = list(range(N)) if phase == "closing" else open_idx
        targets = phase_targets(panel, awake, seasons, anchor, phase)

        print(f"\n{'#' * 70}")
        print(f"ANCHOR {anchor} ({phase}), coverage {awake[:, anchor_col].mean() * 100:.1f}% "
              f"-- mixture panel: {len(panel_idx)} bookmakers minus the target")
        print(f"{'#' * 70}")
        for name, cov, ns in targets:
            print(f"  target {name:8s} coverage {cov:5.1f}%  seasons {ns}")
        print()

        for name, cov, _ in targets:
            t_col = panel.index(name)
            odds_full = load_decimal_odds(name, match_ids)

            # common opportunity set: anchor, target and odds all available
            common = awake[:, anchor_col] & awake[:, t_col] & ~np.isnan(odds_full[:, 0])
            n_common = int(common.sum())
            print(f"--- {name}: {n_common} rounds with {anchor} + {name} + odds ---")

            y_c, odds_c = y[common], odds_full[common]
            all_true = np.ones(n_common, dtype=bool)

            # forecaster 1/2: the anchor's own price
            anc_raw = P[common][:, anchor_col, :]
            _, anc_cal, _, _ = calibrate_full_history(P[:, anchor_col, :], y, common)

            # forecaster 3/4: the mixture on the admissible panel, target excluded,
            # fitted on ALL rounds and then bet only on `common`
            keep_cols = [k for k in panel_idx if k != t_col]
            P_sub, awake_sub = P[:, keep_cols, :], awake[:, keep_cols]
            W = run_ogd_sleeping(per_round_expert_loss(P_sub, y), awake_sub, len(keep_cols))
            ogd_full = np.einsum("tn,tnk->tk", W, P_sub)
            _, ogd_cal_full, _, _ = calibrate_full_history(
                ogd_full, y, np.ones(len(y), dtype=bool))
            ogd_raw, ogd_cal = ogd_full[common], ogd_cal_full[common]

            growth_by = {}
            for fname, p_hat in [("anchor raw", anc_raw), ("anchor calibrated", anc_cal),
                                  ("OGD raw", ogd_raw), ("OGD calibrated", ogd_cal)]:
                growth, bet_placed = simulate_kelly_betting(p_hat, odds_c, y_c, all_true)
                row, _ = score_bet(growth, bet_placed, all_true,
                                    {"anchor": anchor, "phase": phase, "bookmaker": name,
                                     "forecaster": fname, "target_coverage_pct": cov})
                row["thin"] = row["n_bets"] < 30
                rows.append(row)
                growth_by[fname] = np.log(growth)   # 0 on no-bet rounds, per common round
                print(f"  [{fname:18s}] bets={row['n_bets']:6d} "
                      f"({row['bet_rate_pct']:.1f}%)  bankroll={row['final_bankroll']:.4f}  "
                      f"edge={row['mean_log_growth_per_bet']:+.5f}  p={row['p_value']}")

            # paired, per active round (the four bet different rounds)
            for a, b in [("anchor calibrated", "OGD calibrated"),
                          ("anchor raw", "OGD raw"),
                          ("anchor calibrated", "anchor raw"),
                          ("OGD calibrated", "OGD raw")]:
                diff = growth_by[a] - growth_by[b]
                res = moving_block_bootstrap_test(diff)
                verdict = ("tie (not significant)" if not res["significant_95"]
                           else (f"{a} better" if res["mean_diff"] > 0 else f"{b} better"))
                sig_rows.append({"anchor": anchor, "phase": phase, "bookmaker": name,
                                  "comparison": f"{a} vs {b}", "n_rounds": n_common,
                                  "verdict": verdict, **res})
            print()

    table = pd.DataFrame(rows)
    table_path = os.path.join(OUT_DIR, "value_betting_sharp_anchor_table.csv")
    table.to_csv(table_path, index=False)

    sig = pd.DataFrame(sig_rows)
    sig_path = os.path.join(OUT_DIR, "value_betting_sharp_anchor_significance.csv")
    sig.to_csv(sig_path, index=False)

    pd.set_option("display.width", 210)
    print(f"\n=== Results — saved to {table_path} ===\n")
    for anchor, phase in ANCHORS:
        sub = table[table["anchor"] == anchor]
        print(f"-- anchor {anchor} ({phase}) --")
        print(sub.pivot(index="bookmaker", columns="forecaster",
                         values=["n_bets", "final_bankroll", "mean_log_growth_per_bet"])
              .to_string(float_format=lambda v: f"{v:.4f}"))
        print()

    print(f"\n=== Paired, per active round — saved to {sig_path} ===")
    print("    (POSITIVE mean_diff = the FIRST-named forecaster has the higher growth)\n")
    head = sig[sig["comparison"] == "anchor calibrated vs OGD calibrated"]
    for anchor, phase in ANCHORS:
        sub = head[head["anchor"] == anchor]
        k = int((sub["mean_diff"] > 0).sum())
        print(f"-- {anchor} calibrated vs OGD calibrated ({phase}) --")
        print(sub[["bookmaker", "n_rounds", "mean_diff", "ci_low", "ci_high",
                    "p_value", "verdict"]]
              .to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        print(f"   {anchor} ahead in {k} of {len(sub)} targets; "
              f"significant in {int(sub['significant_95'].sum())}.\n")

    plot_anchor(table, os.path.join(OUT_DIR, "value_betting_sharp_anchor.png"))


if __name__ == "__main__":
    main()
