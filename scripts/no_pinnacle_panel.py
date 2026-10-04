"""
HOW GOOD IS THE MIXTURE WITHOUT PINNACLE IN IT?

Pinnacle is the elephant in this panel. PSC is the best single forecaster in
the project, PS is a strong opening price, and every result about the
mixture is shadowed by the fact that Pinnacle is one of the experts being
mixed: "the mixture beats every bookmaker except PSC" invites the obvious
reply that it is mostly reproducing PSC.

This file removes BOTH Pinnacle columns (PS and PSC) from the panel
entirely, refits all four algorithms from scratch on what is left, and ranks
them against the remaining individual bookmakers on LOG-LOSS. The question
it answers: without access to the sharpest odds-setter in the market, does
the learned mixture still beat the bookmakers it can see?

STANDALONE ON PURPOSE. It reads the same processed data as everything else
but writes only its own outputs and changes nothing in the main pipeline --
the panel used everywhere else keeps all 26 experts. Run it whenever; it
depends on no other script's results.

Comparisons are ranked on raw and calibrated log-loss, plus a paired
moving block bootstrap of the best algorithm against each bookmaker, on the
rounds where both are defined. SIGN: diff = loss(OGD) - loss(bookmaker), so
NEGATIVE means OGD is better (the log-loss convention, see CLAUDE.md).

Produces:
  - results/no_pinnacle_ranking.csv
  - results/no_pinnacle_significance.csv
  - results/no_pinnacle.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    OUT_DIR, EPS, COLORS,
    load_full_universe, per_round_expert_loss,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping, evaluate,
)
from final_ranking import calibrate_full_history, COVERAGE_THRESHOLD, RANK_COLORS
from significance_test import moving_block_bootstrap_test, paired_diff, full_length_raw_logloss

DROPPED = ["PS", "PSC"]          # both Pinnacle columns
CHALLENGER = "OGD"               # best algorithm, the one the tests are run for
TRAIN_FRACTION = 0.75            # chronological split, same protocol as elsewhere

# Run on two panels. The closing-only one is there because
# best_configuration.py found it to be the project's best configuration --
# significantly better than the full 26-expert panel (-0.00034, p<0.001) and
# statistically TIED with PSC, where the full panel loses to it significantly.
# Its coverage floor has to be lower: closing columns only exist from ~2019/20
# for most bookmakers, so once PSC is removed NOTHING in that panel reaches
# the 70% used elsewhere (the next best, B365C, sits at 69.6%).
# The opening panel is there for symmetry: PS and PSC are the same company, so
# "without Pinnacle" has to mean dropping it from whichever phase is on the
# table, not only from the closing side.
PANELS = [
    ("full panel (24 experts)", None, 70.0),
    ("closing-only panel (12 experts)", "closing", 40.0),
    ("opening-only panel (12 experts)", "opening", 40.0),
]


def plot_no_pinnacle(table, out_path, label="full panel"):
    """Slope chart, raw -> calibrated log-loss, one row per surviving entity,
    same convention as final_ranking.plot_slope."""
    df = table.sort_values("calibrated_log_loss").reset_index(drop=True)
    n = len(df)
    fig, ax = plt.subplots(figsize=(9, 0.5 * n + 2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ypos = np.arange(n)[::-1]
    for yp, (_, r) in zip(ypos, df.iterrows()):
        color = RANK_COLORS[r["type"]]
        ax.plot([r["raw_log_loss"], r["calibrated_log_loss"]], [yp, yp],
                 color=color, linewidth=1.6, zorder=2, alpha=0.9)
        ax.scatter([r["raw_log_loss"]], [yp], s=70, facecolor="#fcfcfb",
                    edgecolor=color, linewidth=1.8, zorder=3)
        ax.scatter([r["calibrated_log_loss"]], [yp], s=70, facecolor=color,
                    edgecolor=color, linewidth=1.0, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(df["name"], fontsize=9, color="#0b0b0b")
    ax.set_xlabel("Log-loss (full causal history)", color="#52514e")
    ax.set_title(f"Pinnacle removed -- {label}\nopen = raw, filled = calibrated",
                  color="#0b0b0b", fontsize=12, pad=12)
    ax.grid(True, axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", left=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def run_panel(label, restrict, threshold, panel, P, awake, y, seasons, suffix):
    """One panel with Pinnacle removed: refit everything, rank, test.
    `restrict=None` keeps all 26 columns, `restrict="closing"` keeps only the
    closing side first (the best configuration found by
    best_configuration.py)."""
    T, N_full, _ = P.shape
    base = list(range(N_full))
    if restrict == "closing":
        base = [k for k, name in enumerate(panel) if name.endswith("C")]
    elif restrict == "opening":
        base = [k for k, name in enumerate(panel) if not name.endswith("C")]

    keep = [k for k in base if panel[k] not in DROPPED]
    sub_panel = [panel[k] for k in keep]
    P_s, awake_s = P[:, keep, :], awake[:, keep]
    N = len(sub_panel)
    print(f"\n{'#' * 70}\n{label} -- dropping {DROPPED} -> {N} experts\n{'#' * 70}")
    print(f"{sub_panel}")

    # every round must still have at least one quote once Pinnacle is gone
    alive = awake_s.any(axis=1)
    print(f"rounds with >=1 remaining bookmaker: {int(alive.sum())} of {T} "
          f"({100 * alive.mean():.2f}%)\n")

    loss = per_round_expert_loss(P_s, y)
    W = {"Hedge": run_hedge_sleeping(loss, awake_s, N),
         "OGD": run_ogd_sleeping(loss, awake_s, N),
         "FTRL": run_ftrl_sleeping(loss, awake_s, N),
         "Uniform average": np.divide(awake_s, awake_s.sum(axis=1, keepdims=True),
                                       out=np.zeros_like(awake_s, dtype=float),
                                       where=awake_s.sum(axis=1, keepdims=True) > 0)}

    entries = {}
    for name, Wm in W.items():
        entries[name] = (np.einsum("tn,tnk->tk", Wm, P_s), alive, "algorithm")
    coverage = awake_s.mean(axis=0) * 100
    for k, book in enumerate(sub_panel):
        n_seasons = len(set(seasons[awake_s[:, k]]))
        if coverage[k] >= threshold and n_seasons > 1:
            entries[book] = (P_s[:, k, :], awake_s[:, k], "bookmaker")

    # ---- per-round losses, then ONE common round set for everybody ----
    # Same discipline as best_configuration.py: series with different coverage
    # cannot be ranked against each other on their own private match sets, and
    # a configuration picked by looking at these numbers needs a held-out tail.
    per_round = {}
    for name, (probs, mask, kind) in entries.items():
        _, cal, yy, _ = calibrate_full_history(probs, y, mask)
        raw = full_length_raw_logloss(probs, y, T, mask=mask)
        cal_full = np.full(T, np.nan)
        cal_full[mask] = -np.log(np.clip(cal[np.arange(len(yy)), yy], EPS, 1.0))
        per_round[name] = {"raw": raw, "calibrated": cal_full, "type": kind,
                            "coverage_pct": mask.mean() * 100}

    common = np.ones(T, dtype=bool)
    for s in per_round.values():
        common &= ~np.isnan(s["raw"]) & ~np.isnan(s["calibrated"])
    n_common = int(common.sum())
    idx_common = np.flatnonzero(common)
    n_train = int(round(n_common * TRAIN_FRACTION))
    tr = np.zeros(T, dtype=bool); tr[idx_common[:n_train]] = True
    val = np.zeros(T, dtype=bool); val[idx_common[n_train:]] = True
    print(f"common rounds (every series defined): {n_common} "
          f"-- train {n_train} / held-out {n_common - n_train}\n")

    rows = []
    for name, s in per_round.items():
        rows.append({
            "panel": label, "name": name, "type": s["type"],
            "coverage_pct": s["coverage_pct"], "common_rounds": n_common,
            "raw_log_loss": float(np.nanmean(np.where(common, s["raw"], np.nan))),
            "calibrated_log_loss": float(np.nanmean(np.where(common, s["calibrated"], np.nan))),
            "train_log_loss": float(np.nanmean(np.where(tr, s["calibrated"], np.nan))),
            "val_log_loss": float(np.nanmean(np.where(val, s["calibrated"], np.nan))),
        })

    table = pd.DataFrame(rows).sort_values("calibrated_log_loss").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))

    pd.set_option("display.width", 180)
    print(f"=== Ranking WITHOUT Pinnacle -- {label}, ALL SERIES ON THE SAME {n_common} ROUNDS ===")
    print(f"    (bookmakers: coverage >= {threshold:.0f}%, multi-season)\n")
    print(table.drop(columns=["panel", "common_rounds"])
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    best_book = table[table["type"] == "bookmaker"].iloc[0]
    algo_rank = int(table[table["name"] == CHALLENGER]["rank"].iloc[0])
    print(f"\n{CHALLENGER} ranks #{algo_rank} of {len(table)}; "
          f"best surviving bookmaker is {best_book['name']} at #{int(best_book['rank'])}.")
    val_order = table.sort_values("val_log_loss")
    print(f"on the held-out tail the order starts: "
          f"{', '.join(val_order['name'].head(3))}")

    # ---- paired tests on those same common rounds ----
    sig_rows = []
    for name, s in per_round.items():
        if s["type"] != "bookmaker":
            continue
        a = np.where(common, per_round[CHALLENGER]["raw"], np.nan)
        b = np.where(common, s["raw"], np.nan)
        d, n = paired_diff(a, b)
        res = moving_block_bootstrap_test(d)
        verdict = ("tie (not significant)" if not res["significant_95"]
                   else (f"{CHALLENGER} better" if res["mean_diff"] < 0 else f"{name} better"))
        sig_rows.append({"bookmaker": name, "n": n, "verdict": verdict, **res})

    sig = pd.DataFrame(sig_rows).sort_values("mean_diff")
    sig.insert(0, "panel", label)
    print(f"\n=== {CHALLENGER} vs each surviving bookmaker, raw log-loss, paired ===")
    print(f"    (negative mean_diff = {CHALLENGER} better)\n")
    print(sig[["bookmaker", "n", "mean_diff", "ci_low", "ci_high", "p_value", "verdict"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    wins = int((sig["verdict"] == f"{CHALLENGER} better").sum())
    losses = int((sig["verdict"].str.endswith("better") &
                  (sig["verdict"] != f"{CHALLENGER} better")).sum())
    print(f"\n{CHALLENGER} significantly beats {wins} of {len(sig)} surviving bookmakers; "
          f"loses to {losses}.")

    plot_no_pinnacle(table, os.path.join(OUT_DIR, f"no_pinnacle{suffix}.png"), label)
    return table, sig


def main():
    panel, P, awake, y, dates, seasons = load_full_universe()

    tables, sigs = [], []
    for (label, restrict, threshold), suffix in zip(PANELS, ["", "_closing", "_opening"]):
        t, s = run_panel(label, restrict, threshold, panel, P, awake, y, seasons, suffix)
        tables.append(t)
        sigs.append(s)

    out = os.path.join(OUT_DIR, "no_pinnacle_ranking.csv")
    sig_path = os.path.join(OUT_DIR, "no_pinnacle_significance.csv")
    pd.concat(tables, ignore_index=True).to_csv(out, index=False)
    pd.concat(sigs, ignore_index=True).to_csv(sig_path, index=False)
    print(f"\nSaved: {out}\nSaved: {sig_path}")


if __name__ == "__main__":
    main()
