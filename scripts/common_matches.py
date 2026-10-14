"""
COMMON MATCHES ONLY: THE PANEL WITHOUT PARTIAL COVERAGE.

Every other result scores a panel in which bookmakers appear and disappear,
handled by the Sleeping Experts reduction. Two questions that leaves open:
are the findings partly an artefact of partial coverage, and is Betfair
Exchange (which beats us in the deployment test) a better forecaster than
Pinnacle, or only the best one left once Pinnacle stopped quoting?

So this file builds datasets of matches that EVERY chosen series quoted. Each
expert is then awake every round and Hedge / OGD / FTRL are the classical
algorithms (the sleeping implementations reduce to them exactly when nobody
sleeps). The series were chosen greedily for the largest intersection:

  B   12 series, 6 opening + 6 closing, 31,596 matches, 2019/20-2023/24
  B'  the same matches without Pinnacle (PS, PSC): 10 series
  C   10 series including Betfair Exchange, 6,941 matches, 2024/25

Each algorithm's step multiplier is re-selected on the dataset's chronological
75% prefix: with nobody asleep the thesis's multipliers are no longer the right
ones (Hedge and FTRL want steps hundreds of times larger). Every series is
temperature-scaled the project's way, and every comparison is a paired moving
block bootstrap, over the whole dataset and over its 25% tail. All series are
scored on the same matches, so their log-losses compare directly.

Value betting repeats value_betting.py's design on B and B': quarter Kelly
against the target's real odds, OGD mixture of the target's phase with the
target removed (an opening target sees no closing price at all).

Produces:
  - results/common_matches_table.csv          (calibrated log-loss per series)
  - results/common_matches_significance.csv   (paired tests)
  - results/common_matches_value_betting.csv  (one row per panel x target)
"""

import os
import numpy as np
import pandas as pd

import learning_rate_study as lr
from sleeping_experts import OUT_DIR, EPS, load_full_universe, load_match_order, per_round_expert_loss
from final_ranking import calibrate_full_history
from significance_test import moving_block_bootstrap_test
from betting import load_decimal_odds, simulate_kelly_betting, score_bet

TRAIN_FRACTION = 0.75
B_OPENING = ["B365", "BW", "PS", "WH", "VC_BV", "IW"]
B_CLOSING = ["B365C", "BWC", "PSC", "WHC", "VC_BVC", "IWC"]
C_SERIES = ["1XB", "1XBC", "B365", "B365C", "BF", "BFC", "BFE", "BFEC", "PS", "PSC"]
NO_PINNACLE = ("PS", "PSC")

# Step-multiplier grids. With every expert awake, Hedge and FTRL select steps
# far above the thesis's; the grids reach far enough that the selection is
# interior or sits on a plateau where the algorithm simply follows the best
# expert (larger steps change nothing further).
GRIDS = {
    "Hedge": (lr.hedge_scaled, [0.03, 0.1, 0.25, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]),
    "OGD": (lr.ogd_scaled, [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]),
    "FTRL": (lr.ftrl_scaled, [0.01, 0.05, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]),
}

# (first, second) pairs tested per dataset; negative mean diff = first better
TESTS = {
    "B": [("OGD", "PSC"), ("FTRL", "PSC"), ("Hedge", "PSC"), ("OGD", "VC_BVC"),
          ("OGD", "Uniform average"), ("FTRL", "Uniform average"), ("Hedge", "Uniform average")],
    "B'": [("OGD", "VC_BVC"), ("FTRL", "VC_BVC"), ("Hedge", "VC_BVC"), ("OGD", "B365C"),
           ("OGD", "Uniform average"), ("FTRL", "Uniform average"), ("Hedge", "Uniform average"),
           ("Uniform average", "VC_BVC")],
    "C": [("PSC", "BFEC"), ("OGD", "PSC"), ("FTRL", "PSC"), ("OGD", "BFEC"), ("FTRL", "BFEC"),
          ("OGD", "Uniform average")],
}


class Dataset:
    """The matches every series in `defining` quoted, panel restricted to `books`."""

    def __init__(self, universe, defining, books):
        panel, P_all, awake_all, y_all, dates, seasons = universe
        self.rows = awake_all[:, [panel.index(b) for b in defining]].all(axis=1)
        self.y = y_all[self.rows]
        self.T = len(self.y)
        self.n_train = int(round(self.T * TRAIN_FRACTION))
        self.panel = panel
        self.P_all = P_all
        self.seasons = sorted(set(seasons[self.rows]))
        self.books = books

    def probs(self, books):
        return self.P_all[self.rows][:, [self.panel.index(b) for b in books], :]

    def ll(self, p):
        return -np.log(np.clip(p[np.arange(self.T), self.y], EPS, 1.0))

    def calibrated(self, p):
        _, c, _, _ = calibrate_full_history(p, self.y, np.ones(self.T, dtype=bool))
        return c

    def mixture(self, books, algo):
        """Classical algorithm on `books`, step selected on the prefix (raw log-loss)."""
        P = self.probs(books)
        N = len(books)
        awake = np.ones((self.T, N), dtype=bool)
        loss = per_round_expert_loss(P, self.y)
        fn, grid = GRIDS[algo]

        def mix(m):
            return np.einsum("tn,tnk->tk", fn(loss, awake, N, mult=m), P)

        m = min(grid, key=lambda m: self.ll(mix(m))[:self.n_train].mean())
        return mix(m), m


def score_dataset(name, d):
    """Calibrated per-round log-loss of every algorithm, the average and each series."""
    losses, rows = {}, []
    for algo in GRIDS:
        p, step = d.mixture(d.books, algo)
        losses[algo] = d.ll(d.calibrated(p))
        rows.append({"dataset": name, "series": algo, "type": "algorithm", "step_multiplier": step})
    losses["Uniform average"] = d.ll(d.calibrated(d.probs(d.books).mean(axis=1)))
    rows.append({"dataset": name, "series": "Uniform average", "type": "baseline"})
    for b in d.books:
        losses[b] = d.ll(d.calibrated(d.probs([b])[:, 0, :]))
        rows.append({"dataset": name, "series": b, "type": "bookmaker",
                     "phase": "closing" if b.endswith("C") else "opening"})
    for r in rows:
        l = losses[r["series"]]
        r.update({"matches": d.T, "seasons": f"{d.seasons[0]}-{d.seasons[-1]}",
                  "calibrated_log_loss": l.mean(), "tail_log_loss": l[d.n_train:].mean()})
    return rows, losses


def paired(name, d, losses):
    out = []
    for a, b in TESTS[name]:
        diff = losses[a] - losses[b]
        r = moving_block_bootstrap_test(diff)
        rt = moving_block_bootstrap_test(diff[d.n_train:])
        out.append({"dataset": name, "comparison": f"{a} vs {b}", "n": d.T,
                    "mean_diff": r["mean_diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                    "p_value": r["p_value"], "significant_95": r["significant_95"],
                    "tail_mean_diff": rt["mean_diff"], "tail_p_value": rt["p_value"]})
    return out


def value_betting(name, d, opening, closing, match_ids):
    """Quarter Kelly against each target's real odds, OGD mixture of the
    target's own phase with the target removed."""
    out = []
    for target in opening + closing:
        peers = [b for b in (opening if target in opening else closing) if b != target]
        p, step = d.mixture(peers, "OGD")
        odds = load_decimal_odds(target, match_ids[d.rows])
        mask = np.isfinite(odds).all(axis=1)
        growth, bet = simulate_kelly_betting(d.calibrated(p), odds, d.y, mask)
        row, _ = score_bet(growth, bet, mask, {
            "panel": name, "bookmaker": target,
            "phase": "closing" if target in closing else "opening",
            "panel_size": len(peers), "step_multiplier": step})
        out.append(row)
        flag = (("profit" if row["mean_log_growth_per_bet"] > 0 else "loss")
                if row["significant_95"] else "-")
        print(f"  {name:3s} {target:7s} bets {row['n_bets']:6d}  bankroll "
              f"{row['final_bankroll']:9.4f}x  p={row['p_value']:.3f}  {flag}")
    return out


def main():
    universe = load_full_universe()
    match_ids = np.asarray(load_match_order())
    b12 = B_OPENING + B_CLOSING
    opening_np = [b for b in B_OPENING if b not in NO_PINNACLE]
    closing_np = [b for b in B_CLOSING if b not in NO_PINNACLE]
    datasets = {
        "B": Dataset(universe, b12, b12),
        "B'": Dataset(universe, b12, opening_np + closing_np),   # same matches as B
        "C": Dataset(universe, C_SERIES, C_SERIES),
    }

    table, sig = [], []
    for name, d in datasets.items():
        rows, losses = score_dataset(name, d)
        table += rows
        sig += paired(name, d, losses)
        print(f"\n=== {name}: {d.T} matches, {d.seasons[0]}-{d.seasons[-1]} ===")
        for r in sorted(rows, key=lambda r: r["calibrated_log_loss"]):
            step = f" (x{r['step_multiplier']:g})" if r["type"] == "algorithm" else ""
            print(f"  {r['series'] + step:22s} {r['calibrated_log_loss']:.5f}  tail {r['tail_log_loss']:.5f}")
        for s in (x for x in sig if x["dataset"] == name):
            print(f"  {s['comparison']:28s} {s['mean_diff']:+.6f} p={s['p_value']:.3f}  "
                  f"| tail {s['tail_mean_diff']:+.6f} p={s['tail_p_value']:.3f}")

    print("\n=== value betting, quarter Kelly ===")
    vb = (value_betting("B", datasets["B"], B_OPENING, B_CLOSING, match_ids)
          + value_betting("B'", datasets["B'"], opening_np, closing_np, match_ids))

    for df, fname in ((pd.DataFrame(table), "common_matches_table.csv"),
                      (pd.DataFrame(sig), "common_matches_significance.csv"),
                      (pd.DataFrame(vb), "common_matches_value_betting.csv")):
        path = os.path.join(OUT_DIR, fname)
        df.to_csv(path, index=False)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
