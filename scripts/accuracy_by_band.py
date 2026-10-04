"""
IS THERE A PROBABILITY RANGE WHERE WE ACTUALLY BEAT THE BOOKMAKERS?

Motivation: every pooled comparison in this project averages over the whole
probability spectrum, and the accuracy tests in market_devig_comparisons.py
(`opening_vs_closing_vs_uniform*.csv`) found accuracy can't separate ANY of
the methods -- 0 significant results out of 18. But a pooled null result can
hide a localised effect: maybe the mixture is indistinguishable on the
favourites (where everyone agrees) and genuinely better on the longshots, or
vice versa. This file looks for exactly that, by stratifying instead of
pooling.

Three views, all restricted to Tier-A entities (4 algorithms + bookmakers
with coverage >= 70%), same cut as final_ranking.py:

  A. RELIABILITY BY PREDICTED-PROBABILITY BAND. Pool all three outcome
     probabilities per match into (predicted, hit) pairs -- the same
     construction as calibration_analysis.py -- bucket them into 10 fixed
     bands (0-10%, 10-20%, ..., 90-100%), and report per band per series:
     how many predictions landed there, the mean predicted probability, the
     realised frequency, and the gap. "Better in the 20-30% band" means a
     smaller |gap| there: the series' stated 25% actually happens ~25% of
     the time.

  B. ARGMAX ACCURACY BY CONFIDENCE BAND. The accuracy question proper: when
     a series' top pick carries 40-50% / 50-60% / ... confidence, how often
     is that pick right? (With 3 outcomes the top pick is always >= 1/3, so
     the bands start at 0.33.) This is the per-interval version of the
     pooled accuracy number in results_table.csv.

  C. IS ANY OF IT REAL? Views A and B are descriptive -- with 10 bands the
     per-band samples get small and differences between series start to look
     impressive by chance. So for each confidence band we also run the
     project's standard paired test: OGD vs each Tier-A bookmaker, per-round
     log-loss, restricted to the rounds where OGD's top pick falls in that
     band (both series defined on those rounds, so the pairing is exact).
     Moving block bootstrap, same engine as everywhere else.

SIGN (view C): diff = loss(OGD) - loss(bookmaker), so NEGATIVE mean_diff
means OGD is better -- the log-loss convention, not the growth convention
(see CLAUDE.md).

Produces:
  - results/accuracy_by_band_reliability.csv   (view A)
  - results/accuracy_by_band_confidence.csv    (view B)
  - results/accuracy_by_band_significance.csv  (view C)
  - results/accuracy_by_band.png               (A and B, one panel each)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    OUT_DIR, EPS, COLORS,
    load_full_universe, per_round_expert_loss,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping,
)
from calibration_analysis import _style_axes
from final_ranking import COVERAGE_THRESHOLD, RANK_COLORS
from significance_test import moving_block_bootstrap_test, paired_diff, full_length_raw_logloss

PROB_EDGES = np.linspace(0.0, 1.0, 11)          # view A: 0-10%, ..., 90-100%
CONF_EDGES = np.array([1 / 3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])  # view B: top-pick confidence
MIN_BAND_N = 200   # bands thinner than this are reported but flagged as unreliable


def band_labels(edges, pct=True):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(f"{lo * 100:.0f}-{hi * 100:.0f}%" if pct else f"{lo:.2f}-{hi:.2f}")
    return out


# ------------------------------------------------- A: reliability by band --

def reliability_by_band(name, probs, y, mask):
    """One row per probability band: how many (match, outcome) predictions the
    series put there, what it claimed on average, and what actually happened."""
    T = probs.shape[0]
    onehot = np.zeros_like(probs)
    onehot[np.arange(T), y] = 1.0
    p_flat = probs[mask].reshape(-1)
    hit_flat = onehot[mask].reshape(-1)

    idx = np.clip(np.digitize(p_flat, PROB_EDGES[1:-1], right=False), 0, len(PROB_EDGES) - 2)
    rows = []
    for b, label in enumerate(band_labels(PROB_EDGES)):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        predicted = float(p_flat[sel].mean())
        actual = float(hit_flat[sel].mean())
        rows.append({
            "series": name, "band": label, "n": n,
            "mean_predicted": predicted, "actual_frequency": actual,
            "gap": actual - predicted, "abs_gap": abs(actual - predicted),
            "thin": n < MIN_BAND_N,
        })
    return rows


# --------------------------------------------- B: accuracy by confidence ---

def accuracy_by_confidence(name, probs, y, mask):
    """One row per confidence band: among the rounds where this series' TOP
    pick carried that much probability, how often was the pick right?"""
    pmax = probs.max(axis=1)
    pick = probs.argmax(axis=1)
    correct = (pick == y)

    idx = np.clip(np.digitize(pmax, CONF_EDGES[1:-1], right=False), 0, len(CONF_EDGES) - 2)
    rows = []
    for b, label in enumerate(band_labels(CONF_EDGES)):
        sel = mask & (idx == b)
        n = int(sel.sum())
        if n == 0:
            continue
        rows.append({
            "series": name, "band": label, "n": n,
            "share_of_rounds_pct": 100.0 * n / int(mask.sum()),
            "mean_confidence": float(pmax[sel].mean()),
            "accuracy": float(correct[sel].mean()),
            "accuracy_minus_confidence": float(correct[sel].mean() - pmax[sel].mean()),
            "thin": n < MIN_BAND_N,
        })
    return rows


# ------------------------------------------------------------- plotting ----

def plot_bands(rel_table, conf_table, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#fcfcfb")
    algos = ["Hedge", "OGD", "FTRL", "Uniform average"]

    ax1.set_facecolor("#fcfcfb")
    ax1.axhline(0, color="#898781", linewidth=1.4, linestyle="--", zorder=1)
    for series, g in rel_table[~rel_table["thin"]].groupby("series"):
        is_algo = series in algos
        ax1.plot(g["mean_predicted"], g["gap"] * 100,
                  color=COLORS.get(series, RANK_COLORS["bookmaker"]) if is_algo else "#c3c2b7",
                  linewidth=2.0 if is_algo else 1.1, alpha=1.0 if is_algo else 0.75,
                  marker="o", markersize=4.5 if is_algo else 3,
                  zorder=3 if is_algo else 2, label=series if is_algo else None)
    ax1.set_xlabel("Mean predicted probability in band", color="#52514e")
    ax1.set_ylabel("Actual − predicted  (percentage points)", color="#52514e")
    ax1.set_title("A. Reliability by probability band\n(algorithms coloured, bookmakers grey)",
                   color="#0b0b0b", fontsize=12, pad=10)
    _style_axes(ax1)
    leg1 = ax1.legend(frameon=False, loc="best", fontsize=8.5)
    for t in leg1.get_texts():
        t.set_color("#0b0b0b")

    ax2.set_facecolor("#fcfcfb")
    conf_labels = band_labels(CONF_EDGES)
    xpos = np.arange(len(conf_labels))
    for series, g in conf_table[~conf_table["thin"]].groupby("series"):
        is_algo = series in algos
        g = g.set_index("band").reindex(conf_labels)
        ax2.plot(xpos, g["accuracy"] * 100,
                  color=COLORS.get(series, RANK_COLORS["bookmaker"]) if is_algo else "#c3c2b7",
                  linewidth=2.0 if is_algo else 1.1, alpha=1.0 if is_algo else 0.75,
                  marker="o", markersize=4.5 if is_algo else 3,
                  zorder=3 if is_algo else 2, label=series if is_algo else None)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(conf_labels, fontsize=8.5)
    ax2.set_xlabel("Confidence of the top pick", color="#52514e")
    ax2.set_ylabel("Accuracy of the top pick (%)", color="#52514e")
    ax2.set_title("B. Accuracy by confidence band\n(how often the top pick is right)",
                   color="#0b0b0b", fontsize=12, pad=10)
    _style_axes(ax2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


# ---------------------------------------------------------------- main -----

def main():
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    loss = per_round_expert_loss(P, y)

    W = {"Hedge": run_hedge_sleeping(loss, awake, N),
         "OGD": run_ogd_sleeping(loss, awake, N),
         "FTRL": run_ftrl_sleeping(loss, awake, N),
         "Uniform average": awake / awake.sum(axis=1, keepdims=True)}

    entries = {}
    for name, Wm in W.items():
        entries[name] = (np.einsum("tn,tnk->tk", Wm, P), np.ones(T, dtype=bool), "algorithm")
    coverage_pct = awake.mean(axis=0) * 100
    for k, book in enumerate(panel):
        if coverage_pct[k] >= COVERAGE_THRESHOLD:
            entries[book] = (P[:, k, :], awake[:, k], "bookmaker")
    print(f"Tier-A entities: {list(entries)}\n")

    rel_rows, conf_rows = [], []
    for name, (probs, mask, kind) in entries.items():
        rel_rows += [{**r, "type": kind} for r in reliability_by_band(name, probs, y, mask)]
        conf_rows += [{**r, "type": kind} for r in accuracy_by_confidence(name, probs, y, mask)]

    rel_table = pd.DataFrame(rel_rows)
    conf_table = pd.DataFrame(conf_rows)
    rel_path = os.path.join(OUT_DIR, "accuracy_by_band_reliability.csv")
    conf_path = os.path.join(OUT_DIR, "accuracy_by_band_confidence.csv")
    rel_table.to_csv(rel_path, index=False)
    conf_table.to_csv(conf_path, index=False)

    pd.set_option("display.width", 200)
    print(f"=== A. Reliability by probability band — saved to {rel_path} ===")
    print("   (gap = actual - predicted; smaller |gap| = the stated probability is more honest)\n")
    for band in band_labels(PROB_EDGES):
        sub = rel_table[(rel_table["band"] == band) & (~rel_table["thin"])]
        if sub.empty:
            continue
        sub = sub.sort_values("abs_gap")
        best_algo = sub[sub["type"] == "algorithm"].head(1)
        best_book = sub[sub["type"] == "bookmaker"].head(1)
        winner = sub.iloc[0]
        print(f"-- band {band}  (winner: {winner['series']}, |gap|={winner['abs_gap'] * 100:.2f}pp) --")
        if not best_algo.empty and not best_book.empty:
            a, b = best_algo.iloc[0], best_book.iloc[0]
            print(f"   best algorithm: {a['series']:16s} gap={a['gap'] * 100:+.2f}pp  n={a['n']:7d}")
            print(f"   best bookmaker: {b['series']:16s} gap={b['gap'] * 100:+.2f}pp  n={b['n']:7d}")

    print(f"\n\n=== B. Accuracy by confidence band — saved to {conf_path} ===\n")
    for band in band_labels(CONF_EDGES):
        sub = conf_table[(conf_table["band"] == band) & (~conf_table["thin"])]
        if sub.empty:
            continue
        sub = sub.sort_values("accuracy", ascending=False)
        print(f"-- confidence {band} --")
        print(sub[["series", "type", "n", "share_of_rounds_pct", "mean_confidence", "accuracy"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print()

    # ---- C: is any apparent per-band advantage statistically real? ----
    print("\n=== C. OGD vs each Tier-A bookmaker, WITHIN each confidence band ===")
    print("    (rounds where OGD's top pick falls in that band; negative mean_diff = OGD better)\n")
    ogd_probs = entries["OGD"][0]
    ogd_pmax = ogd_probs.max(axis=1)
    band_idx = np.clip(np.digitize(ogd_pmax, CONF_EDGES[1:-1], right=False), 0, len(CONF_EDGES) - 2)
    ogd_ll = full_length_raw_logloss(ogd_probs, y, T)

    sig_rows = []
    for b, band in enumerate(band_labels(CONF_EDGES)):
        in_band = band_idx == b
        if in_band.sum() < MIN_BAND_N:
            continue
        for name, (probs, mask, kind) in entries.items():
            if kind != "bookmaker":
                continue
            book_ll = full_length_raw_logloss(probs, y, T, mask=mask)
            a = np.where(in_band, ogd_ll, np.nan)
            b_ll = np.where(in_band, book_ll, np.nan)
            d, n = paired_diff(a, b_ll)
            if n < MIN_BAND_N:
                continue
            res = moving_block_bootstrap_test(d)
            verdict = ("tie (not significant)" if not res["significant_95"]
                       else ("OGD better" if res["mean_diff"] < 0 else f"{name} better"))
            sig_rows.append({"band": band, "bookmaker": name, "n": n,
                              "verdict": verdict, **res})

    sig_table = pd.DataFrame(sig_rows)
    sig_path = os.path.join(OUT_DIR, "accuracy_by_band_significance.csv")
    sig_table.to_csv(sig_path, index=False)
    for band in band_labels(CONF_EDGES):
        sub = sig_table[sig_table["band"] == band]
        if sub.empty:
            continue
        print(f"-- confidence {band} --")
        print(sub[["bookmaker", "n", "mean_diff", "ci_low", "ci_high", "p_value", "verdict"]]
              .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        print()
    print(f"Saved: {sig_path}")

    wins = sig_table[sig_table["verdict"] == "OGD better"]
    print(f"\nOGD significantly better in {len(wins)} of {len(sig_table)} band x bookmaker cells.")
    if not wins.empty:
        print(wins[["band", "bookmaker", "mean_diff", "p_value"]]
              .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    plot_bands(rel_table, conf_table, os.path.join(OUT_DIR, "accuracy_by_band.png"))


if __name__ == "__main__":
    main()
