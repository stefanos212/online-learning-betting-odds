"""
WHERE DOES THE CLOSING-ODDS ADVANTAGE ACTUALLY COME FROM?

market_devig_comparisons.py established that a closing-only mixture beats an
opening-only one by ~0.0034 log-loss, for all four algorithms, and read that
as "the market absorbs information between open and kickoff". That is an
AVERAGE. This file asks whether it is uniform across matches or concentrated
in the minority of matches where the line actually moved -- i.e. whether the
average is hiding a mechanism.

TWO STRATIFIERS, both already derivable from data the project has:

  1. LINE MOVEMENT. For every bookmaker with both an opening and a closing
     column (13 pairs), the total-variation distance between its opening and
     closing probability vector, averaged over the pairs available for that
     match. Large value = the price moved a lot between posting and kickoff,
     which is what information arriving looks like from outside.

  2. SHIN'S z. data_processer.py already fits Shin's "insider-trading
     proportion" per match per bookmaker (column `shin_z` in
     odds_long_shin.csv) -- it has only ever been used as an intermediate
     step to get probabilities, never analysed as a signal. Shin's model
     says the bookmaker widens its margin to protect itself against informed
     bettors, so a high z is the bookmaker's own estimate of how exposed it
     feels on that match.
     Measured as an ANOMALY within bookmaker (z minus that bookmaker's own
     mean), because z tracks the bookmaker's overround almost exactly, so
     the raw value would mostly encode which bookmakers happened to quote.

WHAT THIS CAN AND CANNOT SHOW. Odds alone cannot separate INSIDER
information from PUBLIC information: a big move may be a leaked lineup or
simply an announced injury. The honest claim is about information ARRIVING,
not about its nature. The respectable framing for this is market
microstructure -- informed vs uninformed traders, spread as protection
(Kyle 1985; Glosten & Milgrom 1985; Shin 1992 for the betting-market case)
-- not "insider detection".

Only OGD and the uniform average are fitted here (not Hedge/FTRL): the
question is about the market, not about which algorithm wins, and halving
the fits keeps this cheap.

Produces:
  - results/information_arrival_strata.csv
  - results/information_arrival_significance.csv
  - results/information_arrival.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    DATA_DIR, OUT_DIR, EPS, COLORS,
    load_full_universe, per_round_expert_loss, run_ogd_sleeping,
    load_match_order, split_by_market_phase,
)
from calibration_analysis import _style_axes
from market_devig_comparisons import (
    build_subset,
)
from significance_test import moving_block_bootstrap_test, paired_diff

N_STRATA = 5          # quintiles
MIN_PAIRS = 3         # a match needs this many opening/closing pairs to be scored


def movement_per_match(match_ids):
    """Mean total-variation distance between each bookmaker's opening and
    closing probability vector, over the pairs available for that match.
    Returns a Series aligned to `match_ids` (NaN where too few pairs)."""
    long = pd.read_csv(os.path.join(DATA_DIR, "odds_long.csv"),
                        usecols=["MatchID", "Bookmaker", "pH", "pD", "pA"])
    piv = long.pivot(index="MatchID", columns="Bookmaker", values=["pH", "pD", "pA"])
    books = sorted(long["Bookmaker"].unique())
    pairs = [(b, b + "C") for b in books if not b.endswith("C") and b + "C" in books]
    print(f"opening/closing pairs used for movement: {len(pairs)} -- {[p[0] for p in pairs]}")

    tv_sum = pd.Series(0.0, index=piv.index)
    tv_cnt = pd.Series(0, index=piv.index)
    for op, cl in pairs:
        d = sum((piv[(k, cl)] - piv[(k, op)]).abs() for k in ["pH", "pD", "pA"]) / 2.0
        ok = d.notna()
        tv_sum[ok] += d[ok]
        tv_cnt[ok] += 1
    mov = (tv_sum / tv_cnt).where(tv_cnt >= MIN_PAIRS)
    return mov.reindex(match_ids), tv_cnt.reindex(match_ids)


def z_anomaly_per_match(match_ids):
    """Shin's z per match, as a within-bookmaker anomaly averaged over the
    bookmakers that quoted. Raw z tracks each bookmaker's overround, so the
    anomaly is what isolates 'unusual for this bookmaker, on this match'."""
    long = pd.read_csv(os.path.join(DATA_DIR, "odds_long_shin.csv"),
                        usecols=["MatchID", "Bookmaker", "shin_z"])
    long["z_anom"] = long["shin_z"] - long.groupby("Bookmaker")["shin_z"].transform("mean")
    per_match = long.groupby("MatchID")["z_anom"].mean()
    raw = long.groupby("MatchID")["shin_z"].mean()
    return per_match.reindex(match_ids), raw.reindex(match_ids)


def strata_of(values, n_strata=N_STRATA):
    """Quantile bins, NaN-safe. Returns integer labels (-1 where undefined)."""
    lab = np.full(len(values), -1)
    ok = ~np.isnan(values)
    lab[ok] = pd.qcut(values[ok], n_strata, labels=False, duplicates="drop")
    return lab


def plot_strata(table, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), facecolor="#fcfcfb")
    for ax, stratifier in zip(axes, ["line movement", "Shin z anomaly"]):
        sub = table[table["stratifier"] == stratifier]
        ax.set_facecolor("#fcfcfb")
        x = sub["stratum"].astype(int)
        ax.plot(x, sub["ll_opening"], marker="o", markersize=5, linewidth=1.9,
                 color="#c3c2b7", label="opening-only mixture")
        ax.plot(x, sub["ll_full"], marker="o", markersize=5, linewidth=1.9,
                 color=COLORS["benchmark"], label="full panel")
        ax.plot(x, sub["ll_closing"], marker="o", markersize=5, linewidth=1.9,
                 color=COLORS["OGD"], label="closing-only mixture")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{i+1}" for i in x], fontsize=9)
        ax.set_xlabel(f"quintile of {stratifier} (Q1 = lowest)", color="#52514e")
        ax.set_ylabel("OGD log-loss", color="#52514e")
        ax.set_title(f"Stratified by {stratifier}", color="#0b0b0b", fontsize=12, pad=10)
        _style_axes(ax)
        leg = ax.legend(frameon=False, fontsize=8.5)
        for t in leg.get_texts():
            t.set_color("#0b0b0b")
    fig.suptitle("Does the closing-odds advantage live in the matches where the line moved?",
                  color="#0b0b0b", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")


def main():
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    match_ids = load_match_order()
    assert len(match_ids) == T

    mov, n_pairs = movement_per_match(match_ids)
    z_anom, z_raw = z_anomaly_per_match(match_ids)
    mov, z_anom = mov.values, z_anom.values
    print(f"movement defined for {int((~np.isnan(mov)).sum())} of {T} rounds "
          f"(mean {np.nanmean(mov):.4f}, 90th pct {np.nanpercentile(mov, 90):.4f})")
    print(f"z anomaly defined for {int((~np.isnan(z_anom)).sum())} of {T} rounds\n")

    # OGD on each of the three panels, per-round log-loss scattered to full length
    open_idx, close_idx = split_by_market_phase(panel)
    ll = {}
    for market, indices in [("opening", open_idx), ("closing", close_idx),
                             ("full", list(range(N)))]:
        P_s, aw_s, y_s, keep = build_subset(P, awake, y, indices)
        W = run_ogd_sleeping(per_round_expert_loss(P_s, y_s), aw_s, len(indices))
        phat = np.einsum("tn,tnk->tk", W, P_s)
        v = -np.log(np.clip(phat[np.arange(len(y_s)), y_s], EPS, 1.0))
        full = np.full(T, np.nan)
        full[keep] = v
        ll[market] = full
        print(f"  fitted OGD on {market:8s} panel ({len(indices)} experts, {int(keep.sum())} rounds)")

    rows, sig_rows = [], []
    for stratifier, values in [("line movement", mov), ("Shin z anomaly", z_anom)]:
        lab = strata_of(values)
        print(f"\n=== stratified by {stratifier} ===")
        for s in range(N_STRATA):
            sel = lab == s
            if sel.sum() < 100:
                continue
            row = {"stratifier": stratifier, "stratum": s, "n": int(sel.sum()),
                    "mean_value": float(np.nanmean(values[sel]))}
            for market in ["opening", "closing", "full"]:
                row[f"ll_{market}"] = float(np.nanmean(np.where(sel, ll[market], np.nan)))
            row["closing_minus_opening"] = row["ll_closing"] - row["ll_opening"]
            rows.append(row)

            # paired test within the stratum, on rounds where both are defined
            a = np.where(sel, ll["closing"], np.nan)
            b = np.where(sel, ll["opening"], np.nan)
            d, n = paired_diff(a, b)
            res = moving_block_bootstrap_test(d)
            sig_rows.append({"stratifier": stratifier, "stratum": s, "n": n,
                              "comparison": "closing vs opening", **res})
            print(f"  Q{s+1}: n={int(sel.sum()):6d}  mean={row['mean_value']:+.4f}  "
                  f"opening={row['ll_opening']:.5f} closing={row['ll_closing']:.5f}  "
                  f"diff={res['mean_diff']:+.5f} p={res['p_value']:.4f}")

    table = pd.DataFrame(rows)
    sig = pd.DataFrame(sig_rows)
    tp = os.path.join(OUT_DIR, "information_arrival_strata.csv")
    sp = os.path.join(OUT_DIR, "information_arrival_significance.csv")
    table.to_csv(tp, index=False)
    sig.to_csv(sp, index=False)

    pd.set_option("display.width", 200)
    print(f"\n=== strata — saved to {tp} ===\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\n=== closing vs opening within each stratum — saved to {sp} ===")
    print("    (negative mean_diff = closing better)\n")
    print(sig[["stratifier", "stratum", "n", "mean_diff", "ci_low", "ci_high",
                "p_value", "significant_95"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    for stratifier in table["stratifier"].unique():
        sub = table[table["stratifier"] == stratifier]
        lo, hi = sub.iloc[0], sub.iloc[-1]
        print(f"\n{stratifier}: closing advantage {lo['closing_minus_opening']:+.5f} in Q1 "
              f"-> {hi['closing_minus_opening']:+.5f} in Q{N_STRATA}  "
              f"(ratio {hi['closing_minus_opening'] / lo['closing_minus_opening']:.1f}x)")
        print(f"{stratifier}: full-panel log-loss {lo['ll_full']:.5f} in Q1 "
              f"-> {hi['ll_full']:.5f} in Q{N_STRATA} "
              f"(are these matches simply harder?)")

    plot_strata(table, os.path.join(OUT_DIR, "information_arrival.png"))


if __name__ == "__main__":
    main()
