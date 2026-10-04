"""
The experiment built on sleeping_experts.py's algorithms/data-loading
library: runs Hedge/OGD/FTRL (+ the uniform-average baseline) over the full
26-bookmaker panel via the Sleeping Experts reduction, and reports how they
compare to every individual bookmaker.

Produces:
  - results/results_table.csv           (every algorithm + every bookmaker,
                                           log-loss / Brier / RPS / accuracy)
  - results/sleeping_experts_comparison.png  (two-panel: trend over time +
                                           final ranking dot plot)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    OUT_DIR, EPS, COLORS,
    load_full_universe, per_round_expert_loss,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping, evaluate,
    last_awake_weight,
)

BENCHMARK_MIN_COVERAGE = 0.95   # a benchmark line needs (near-)full coverage to be readable


def cumulative_average(x):
    """Running mean of x, one value per round -- used for the "how does
    average performance evolve over time" plot lines."""
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def cumulative_average_masked(x, mask):
    """Cumulative average over only the masked rounds; holds its last value flat
    on rounds the expert was asleep (so the line stays continuous on a shared
    round axis instead of showing gaps)."""
    x_masked = np.where(mask, x, 0.0)
    csum = np.cumsum(x_masked)
    ccount = np.cumsum(mask.astype(float))
    with np.errstate(invalid="ignore"):
        avg = csum / ccount
    return pd.Series(avg).ffill().values


# --------------------------------------------------------------- plot ------

def plot_comparison(algo_curves, benchmark_name, benchmark_curve, table, start=500):
    """Saves results/sleeping_experts_comparison.png: two panels, (A) trend
    over time for the 4 algorithms + a dashed best-bookmaker reference line,
    burn-in trimmed for readability, and (B) a Cleveland dot plot (NOT a bar
    chart -- see the comment below on why) ranking the 4 algorithms plus the
    best/worst individual bookmaker."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.5), facecolor="#fcfcfb",
        gridspec_kw={"width_ratios": [2.1, 1]},
    )

    # ---- Panel A: trend over time (burn-in trimmed for readability) ----
    ax1.set_facecolor("#fcfcfb")
    rounds = np.arange(1, len(benchmark_curve) + 1)
    for name, series in algo_curves.items():
        ax1.plot(rounds[start:], series[start:], label=name, color=COLORS[name], linewidth=2)
    ax1.plot(rounds[start:], benchmark_curve[start:],
              label=f"Best (near-)full-coverage bookmaker ({benchmark_name})",
              color=COLORS["benchmark"], linewidth=2, linestyle="--")

    ax1.set_xlabel("Round (match, chronological)", color="#52514e")
    ax1.set_ylabel("Cumulative average log-loss", color="#52514e")
    ax1.set_title(f"Cumulative average log-loss (rounds {start}+)", color="#0b0b0b", fontsize=12, pad=10)
    ax1.grid(True, color="#e1e0d9", linewidth=0.8)
    ax1.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax1.spines[spine].set_color("#c3c2b7")
    ax1.tick_params(colors="#898781")
    legend = ax1.legend(frameon=False, loc="upper right", fontsize=8.5)
    for text in legend.get_texts():
        text.set_color("#0b0b0b")

    # ---- Panel B: final ranking, Cleveland dot plot ----
    # (a bar's LENGTH implies a meaningful zero; these log-losses all sit near 1.0
    # with tiny differences, so position-encoded dots are the honest form here —
    # a truncated-baseline bar would visually exaggerate the gaps.)
    ax2.set_facecolor("#fcfcfb")
    algo_names = list(algo_curves.keys())
    best_book = table[table["type"] == "bookmaker"].sort_values("log_loss").iloc[0]
    worst_book = table[table["type"] == "bookmaker"].sort_values("log_loss").iloc[-1]

    dots = []
    for name in algo_names:
        dots.append((name, table.loc[table["name"] == name, "log_loss"].values[0], COLORS[name]))
    dots.append((f"Best bookmaker\n({best_book['name']}, {best_book['coverage_pct']:.0f}% cov.)",
                 best_book["log_loss"], COLORS["benchmark"]))
    dots.append((f"Worst bookmaker\n({worst_book['name']}, {worst_book['coverage_pct']:.0f}% cov.)",
                 worst_book["log_loss"], COLORS["muted"]))
    dots.sort(key=lambda b: b[1])

    labels = [b[0] for b in dots]
    values = [b[1] for b in dots]
    colors = [b[2] for b in dots]
    ypos = np.arange(len(dots))[::-1]

    xmin, xmax = min(values), max(values)
    pad = (xmax - xmin) * 0.25
    for yp, v in zip(ypos, values):
        ax2.plot([xmin - pad, xmax + pad], [yp, yp], color="#e1e0d9", linewidth=0.8, zorder=1)
    ax2.scatter(values, ypos, color=colors, s=90, zorder=3, edgecolor="#fcfcfb", linewidth=1.2)
    for yp, v in zip(ypos, values):
        ax2.annotate(f"{v:.4f}", (v, yp), xytext=(0, 9), textcoords="offset points",
                      ha="center", fontsize=7.5, color="#52514e")

    ax2.set_yticks(ypos)
    ax2.set_yticklabels(labels, fontsize=8.5, color="#0b0b0b")
    ax2.set_xlabel("Final avg. log-loss (lower = better)", color="#52514e", fontsize=9)
    ax2.set_title("Final ranking", color="#0b0b0b", fontsize=12, pad=10)
    ax2.set_xlim(xmin - pad, xmax + pad)
    ax2.set_ylim(-0.7, len(dots) - 0.3)
    for spine in ["top", "right", "left"]:
        ax2.spines[spine].set_visible(False)
    ax2.spines["bottom"].set_color("#c3c2b7")
    ax2.tick_params(colors="#898781", left=False)
    ax2.set_xticks([])

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "sleeping_experts_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


# ---------------------------------------------------------------- main -----

def main():
    """Loads the full bookmaker universe, runs all 3 sleeping-experts
    algorithms + the uniform baseline, builds and saves results_table.csv
    (every algorithm + every bookmaker, with the single_season warning
    flag), and saves the comparison plot. This is the main results table
    almost every later script in the project reads or reproduces a variant
    of."""
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    coverage_pct = awake.mean(axis=0) * 100
    all_seasons = sorted(set(seasons))
    print(f"Full bookmaker universe: {N} experts, {T} sequential rounds "
          f"({pd.Timestamp(dates.min()).date()} -> {pd.Timestamp(dates.max()).date()})")
    print(f"Coverage range: {coverage_pct.min():.1f}% ({panel[coverage_pct.argmin()]}) "
          f"to {coverage_pct.max():.1f}% ({panel[coverage_pct.argmax()]})")
    print(f"Seasons in the data: {all_seasons}")

    loss = per_round_expert_loss(P, y)

    W_hedge = run_hedge_sleeping(loss, awake, N)
    W_ogd = run_ogd_sleeping(loss, awake, N)
    W_ftrl = run_ftrl_sleeping(loss, awake, N)
    W_uniform = awake / awake.sum(axis=1, keepdims=True)  # uniform over whoever is awake that round

    algo_weights = {"Hedge": W_hedge, "OGD": W_ogd, "FTRL": W_ftrl, "Uniform average": W_uniform}

    # per-expert Hedge weight the LAST time each one was awake -- NOT W_hedge[-1],
    # which is 0 for every bookmaker that didn't quote the final match (see
    # sleeping_experts.last_awake_weight for why that reading is misleading)
    hedge_weight_last_awake = last_awake_weight(W_hedge, awake)

    rows = []
    algo_curves = {}
    for name, W in algo_weights.items():
        phat = np.einsum("tn,tnk->tk", W, P)
        m = evaluate(phat, y)
        rows.append({
            "name": name, "type": "algorithm", "coverage_pct": 100.0,
            "n_seasons_active": len(all_seasons), "seasons_active": ",".join(all_seasons),
            "single_season": False, "hedge_weight_last_awake": np.nan, **m,
        })
        algo_curves[name] = cumulative_average(-np.log(np.clip(phat[np.arange(T), y], EPS, 1.0)))

    for k, book in enumerate(panel):
        m = evaluate(P[:, k, :], y, mask=awake[:, k])
        seasons_k = sorted(set(seasons[awake[:, k]]))
        rows.append({
            "name": book, "type": "bookmaker", "coverage_pct": coverage_pct[k],
            "n_seasons_active": len(seasons_k), "seasons_active": ",".join(seasons_k),
            "single_season": len(seasons_k) <= 1,
            "hedge_weight_last_awake": hedge_weight_last_awake[k], **m,
        })

    table = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    table_path = os.path.join(OUT_DIR, "results_table.csv")
    table.to_csv(table_path, index=False)

    pd.set_option("display.width", 140)
    print(f"\n=== Results table (sorted by log-loss) — saved to {table_path} ===\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    single_season = table[table["single_season"]].sort_values("log_loss")
    if not single_season.empty:
        print(f"\n=== WARNING: {len(single_season)} bookmakers are active in only ONE season "
              f"(out of {len(all_seasons)}) — their log-loss is confounded with that season's "
              f"difficulty, not a trustworthy skill comparison against multi-season entries ===\n")
        print(single_season[["name", "seasons_active", "coverage_pct", "log_loss"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # benchmark curve: best bookmaker among those with (near-)full coverage
    full_cov = table[(table["type"] == "bookmaker") & (table["coverage_pct"] >= BENCHMARK_MIN_COVERAGE * 100)]
    bench_row = full_cov.sort_values("log_loss").iloc[0]
    bench_idx = panel.index(bench_row["name"])
    bench_loss = -np.log(P[np.arange(T), bench_idx, y])
    bench_curve = cumulative_average_masked(bench_loss, awake[:, bench_idx])

    plot_comparison(algo_curves, bench_row["name"], bench_curve, table)


if __name__ == "__main__":
    main()
