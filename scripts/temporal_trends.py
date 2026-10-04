"""
TEMPORAL TRENDS: is the football-odds market getting more reliable over time?

Motivation: if bookmakers' models genuinely improve across our 10 seasons,
that's important context for interpreting every other result in this
project -- and it's a concrete, TESTABLE justification for using an online
(non-stationarity-robust) learning framework in the first place, not just a
theoretical nicety. If the "right" calibration correction genuinely drifts
season to season, one fixed correction fit across all of them can't fit any
of them as well as a per-season fit could -- the question
calibration_per_season.py / calibration_warmstart.py then test directly.
(An earlier version of this docstring said this explained why the single
GLOBAL correction made things WORSE on the 10-year dataset. It doesn't:
global calibration significantly HELPS. That claim came from misreading
significance_test.py's sign convention -- see CLAUDE.md.)

Four separate looks at the same question, all split by SEASON instead of
pooled across the full history:
  1. log-loss trend per season      -- does prediction quality improve?
  2. overround/margin trend         -- outcome-independent efficiency proxy:
                                        thinner margins <=> more competition
  3. ECE (favorite-longshot bias)   -- does the bias shrink over time?
  4. Mann-Kendall trend test        -- formal significance, not just "the
                                        line looks like it's going down"

Restricted to the 4 bookmakers present in ALL 10 seasons (B365, BW, PS, PSC)
plus the 4 algorithms (always present by construction) -- anyone with
season gaps would make season-to-season comparison confounded with coverage
changes, which is exactly the problem this file exists to avoid.

Produces:
  - results/temporal_trends_table.csv   (season x series x metric, full detail)
  - results/temporal_trends_mk_test.csv (Mann-Kendall test per series x metric)
  - results/temporal_trends.png         (3 panels: log-loss / overround / ECE,
                                          bookmaker-average vs algorithm-average)
"""

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    DATA_DIR, OUT_DIR, EPS,
    load_full_universe, per_round_expert_loss,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping,
)
from calibration_analysis import pooled_prob_outcome_pairs, calibration_table, expected_calibration_error

STABLE_BOOKMAKERS = ["B365", "BW", "PS", "PSC"]  # present in all 10 seasons
TREND_COLORS = {"Bookmaker average": "#eb6834", "Algorithm average": "#2a78d6"}


# --------------------------------------------------------- Mann-Kendall ----

def mann_kendall_test(x):
    """Non-parametric trend test (Mann & Kendall): counts concordant vs.
    discordant pairs across the whole series rather than fitting a line, so
    it doesn't assume linearity or normally-distributed residuals -- a
    better fit than OLS-regression p-values for a short (n=10 seasons)
    series. No ties-correction term (fine here: season-average metrics are
    continuous, ties are essentially impossible)."""
    n = len(x)
    s = sum(np.sign(x[j] - x[k]) for k in range(n - 1) for j in range(k + 1, n))
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))  # two-sided normal-approx p-value
    trend = "increasing" if s > 0 else ("decreasing" if s < 0 else "no trend")
    return {"S": int(s), "Z": z, "p_value": p_value, "trend": trend}


# ------------------------------------------------------------- metrics -----

def per_season_logloss(loss_1d, seasons, season_order, mask=None):
    if mask is None:
        mask = np.ones(len(loss_1d), dtype=bool)
    out = {}
    for s in season_order:
        sel = mask & (seasons == s)
        if sel.any():
            out[s] = float(loss_1d[sel].mean())
    return out


def per_season_ece(probs, y, seasons, season_order, mask=None):
    out = {}
    for s in season_order:
        sel = (seasons == s) if mask is None else (mask & (seasons == s))
        if not sel.any():
            continue
        p_flat, hit_flat = pooled_prob_outcome_pairs(probs, y, mask=sel)
        df = calibration_table(p_flat, hit_flat, s)
        out[s] = expected_calibration_error(df)
    return out


def per_season_overround(season_order):
    """Independent of realized outcomes -- read straight from odds_long.csv."""
    long = pd.read_csv(os.path.join(DATA_DIR, "odds_long.csv"), dtype={"Season": str})
    long = long[long["Bookmaker"].isin(STABLE_BOOKMAKERS)]
    grp = long.groupby(["Bookmaker", "Season"])["overround"].mean().reset_index()
    out = {}
    for book in STABLE_BOOKMAKERS:
        sub = grp[grp["Bookmaker"] == book].set_index("Season")["overround"]
        out[book] = {s: float(sub.get(s, np.nan)) * 100 for s in season_order}
    return out


# --------------------------------------------------------------- plot ------

def plot_trends(table, out_path, season_order):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5), facecolor="#fcfcfb")
    metrics = [("log_loss", "Log-loss"), ("overround_pct", "Overround (%)"), ("ece", "ECE")]
    xpos = np.arange(len(season_order))

    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.set_facecolor("#fcfcfb")
        for series_type in ["Bookmaker average", "Algorithm average"]:
            sub = table[(table["series"] == series_type) & (table["metric"] == metric)]
            sub = sub.set_index("season").reindex(season_order)
            ax.plot(xpos, sub["value"], color=TREND_COLORS[series_type], linewidth=2,
                     marker="o", markersize=5, label=series_type, zorder=3)
        ax.set_xticks(xpos)
        ax.set_xticklabels(season_order, rotation=45, fontsize=8)
        ax.set_ylabel(ylabel, color="#52514e")
        ax.set_title(ylabel, color="#0b0b0b", fontsize=12, pad=10)
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors="#898781")

    legend = axes[0].legend(frameon=False, loc="best", fontsize=8.5)
    for text in legend.get_texts():
        text.set_color("#0b0b0b")
    fig.suptitle("Market reliability over time (10 seasons)", color="#0b0b0b", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")


# ---------------------------------------------------------------- main -----

def main():
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    season_order = sorted(set(seasons))
    loss = per_round_expert_loss(P, y)

    W_hedge = run_hedge_sleeping(loss, awake, N)
    W_ogd = run_ogd_sleeping(loss, awake, N)
    W_ftrl = run_ftrl_sleeping(loss, awake, N)
    W_uniform = awake / awake.sum(axis=1, keepdims=True)
    algo_weights = {"Hedge": W_hedge, "OGD": W_ogd, "FTRL": W_ftrl, "Uniform average": W_uniform}

    rows = []

    # ---- 1 & 3: per-series log-loss and ECE, per season ----
    algo_ll_by_season = {name: [] for name in algo_weights}
    for name, W in algo_weights.items():
        phat = np.einsum("tn,tnk->tk", W, P)
        ll_1d = -np.log(np.clip(phat[np.arange(T), y], EPS, 1.0))
        ll = per_season_logloss(ll_1d, seasons, season_order)
        ece = per_season_ece(phat, y, seasons, season_order)
        for s in season_order:
            rows.append({"series": name, "type": "algorithm", "season": s, "metric": "log_loss", "value": ll.get(s, np.nan)})
            rows.append({"series": name, "type": "algorithm", "season": s, "metric": "ece", "value": ece.get(s, np.nan)})

    book_ll_by_season = {}
    for book in STABLE_BOOKMAKERS:
        k = panel.index(book)
        ll_1d = -np.log(np.clip(P[:, k, :][np.arange(T), y], EPS, 1.0))
        mask = awake[:, k]
        ll = per_season_logloss(ll_1d, seasons, season_order, mask=mask)
        ece = per_season_ece(P[:, k, :], y, seasons, season_order, mask=mask)
        book_ll_by_season[book] = ll
        for s in season_order:
            rows.append({"series": book, "type": "bookmaker", "season": s, "metric": "log_loss", "value": ll.get(s, np.nan)})
            rows.append({"series": book, "type": "bookmaker", "season": s, "metric": "ece", "value": ece.get(s, np.nan)})

    # ---- 2: overround per season (bookmakers only -- algorithms don't quote odds) ----
    overround = per_season_overround(season_order)
    for book in STABLE_BOOKMAKERS:
        for s in season_order:
            rows.append({"series": book, "type": "bookmaker", "season": s, "metric": "overround_pct", "value": overround[book].get(s, np.nan)})

    detail_table = pd.DataFrame(rows)

    # ---- aggregate "Bookmaker average" / "Algorithm average" series for the headline plot ----
    agg_rows = []
    for metric in ["log_loss", "ece", "overround_pct"]:
        for group_name, members in [("Bookmaker average", STABLE_BOOKMAKERS), ("Algorithm average", list(algo_weights))]:
            if metric == "overround_pct" and group_name == "Algorithm average":
                continue  # algorithms don't quote odds -> no margin to report
            sub = detail_table[(detail_table["metric"] == metric) & (detail_table["series"].isin(members))]
            for s in season_order:
                v = sub[sub["season"] == s]["value"].mean()
                agg_rows.append({"series": group_name, "metric": metric, "season": s, "value": v})
    agg_table = pd.DataFrame(agg_rows)

    full_table = pd.concat([detail_table.assign(kind="per-series"), agg_table.assign(kind="aggregate", type="")],
                            ignore_index=True)
    table_path = os.path.join(OUT_DIR, "temporal_trends_table.csv")
    full_table.to_csv(table_path, index=False)

    pd.set_option("display.width", 160)
    print(f"=== Season-level aggregates (saved full per-series detail to {table_path}) ===\n")
    piv = agg_table.pivot_table(index="season", columns=["metric", "series"], values="value")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

    # ---- 4: Mann-Kendall test on every series x metric, plus the 2 aggregates ----
    mk_rows = []
    for (series, metric), g in pd.concat([detail_table, agg_table.assign(type="")]).groupby(["series", "metric"]):
        g = g.set_index("season").reindex(season_order)
        vals = g["value"].values
        if np.isnan(vals).any() or len(vals) < 4:
            continue
        res = mann_kendall_test(vals)
        mk_rows.append({"series": series, "metric": metric, **res})
    mk_table = pd.DataFrame(mk_rows).sort_values(["metric", "series"])
    mk_path = os.path.join(OUT_DIR, "temporal_trends_mk_test.csv")
    mk_table.to_csv(mk_path, index=False)
    print(f"\n=== Mann-Kendall trend test (n=10 seasons) — saved to {mk_path} ===\n")
    print(mk_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    plot_trends(agg_table, os.path.join(OUT_DIR, "temporal_trends.png"), season_order)


if __name__ == "__main__":
    main()
