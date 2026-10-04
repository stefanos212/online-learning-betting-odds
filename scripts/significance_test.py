"""
Moving-block bootstrap significance tests on log-loss differences.

Why block bootstrap and not a plain t-test / i.i.d. bootstrap: consecutive
per-match losses are not independent draws (an upset-heavy or unusually
chalky weekend hits every forecaster's loss at once), so resampling
individual rounds independently would understate the true variance of the
mean difference. The MOVING BLOCK BOOTSTRAP (Kunsch, 1989) resamples
contiguous blocks of rounds instead of single rounds, preserving whatever
short-range serial dependence exists in the loss sequence. Block length
follows the common n^(1/3) rule of thumb (Hall, Horowitz & Jing, 1995).

This is the same idea behind the classic Diebold-Mariano test for comparing
forecasts, just via bootstrap instead of an asymptotic normal approximation
-- no need to assume the loss-differential process is close to Gaussian at
finite T.

For each comparison, "diff" = loss(A) - loss(B) per round, restricted to
rounds where BOTH are defined, in original chronological order. A negative
mean_diff means A has the LOWER (better) log-loss.
"""

import os
import numpy as np
import pandas as pd

from sleeping_experts import (
    OUT_DIR, EPS,
    load_full_universe, per_round_expert_loss,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping,
)
from final_ranking import calibrate_full_history

N_BOOT = 2000
CI_LEVEL = 0.95
SEED = 12345


# ------------------------------------------------------------ bootstrap ----

def moving_block_bootstrap_test(diff, block_length=None, n_boot=N_BOOT, ci_level=CI_LEVEL, seed=SEED):
    n = len(diff)
    if block_length is None:
        block_length = max(1, int(round(n ** (1 / 3))))  # Hall-Horowitz-Jing rule of thumb
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length
    arange_block = np.arange(block_length)

    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = (starts[:, None] + arange_block[None, :]).reshape(-1)[:n]
        boot_means[b] = diff[idx].mean()

    observed = diff.mean()
    alpha = 1 - ci_level
    lo, hi = np.percentile(boot_means, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    p_value = min(1.0, 2 * min((boot_means <= 0).mean(), (boot_means >= 0).mean()))
    return {
        "n": n, "block_length": block_length, "mean_diff": observed,
        "ci_low": lo, "ci_high": hi, "p_value": p_value,
        "significant_95": bool(not (lo <= 0 <= hi)),
    }


# ------------------------------------------------------- per-round losses --

def full_length_raw_logloss(probs, y, T, mask=None):
    """Per-round log-loss for a raw (uncalibrated) series, full T-length,
    NaN-masked wherever `mask` is False (e.g. a bookmaker's asleep rounds)."""
    idx = np.arange(T)
    ll = -np.log(np.clip(probs[idx, y], EPS, 1.0))
    if mask is not None:
        ll = np.where(mask, ll, np.nan)
    return ll


def full_length_calibrated_logloss(probs, y, mask, T):
    """Runs the SAME causal, train-selected temperature calibration as
    final_ranking.py, then scatters the result back to full T-length
    positions (NaN where this series was asleep) so it can be paired against
    any other series on the shared chronological index."""
    p, calibrated, yy, best_c = calibrate_full_history(probs, y, mask)
    idx = np.arange(len(yy))
    ll = -np.log(np.clip(calibrated[idx, yy], EPS, 1.0))
    full = np.full(T, np.nan)
    full[mask] = ll
    return full


def paired_diff(a, b):
    """a, b: two full-T-length per-round loss arrays (possibly with NaN gaps
    at different positions). Returns (a-b) restricted to rounds where BOTH
    are defined, still in original chronological order -- this is the series
    that gets fed into moving_block_bootstrap_test."""
    common = ~np.isnan(a) & ~np.isnan(b)
    return a[common] - b[common], int(common.sum())


# ---------------------------------------------------------------- main -----

def main():
    """Builds raw + calibrated per-round log-loss series for the 4 algorithms
    and 2 reference bookmakers, runs the 12-comparison battery defined below
    (each a call to moving_block_bootstrap_test on a paired_diff), prints and
    saves results/significance_test_table.csv. moving_block_bootstrap_test()
    itself is reused directly by value_betting.py and contextual_experts.py."""
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    loss = per_round_expert_loss(P, y)

    W_hedge = run_hedge_sleeping(loss, awake, N)
    W_ogd = run_ogd_sleeping(loss, awake, N)
    W_ftrl = run_ftrl_sleeping(loss, awake, N)
    W_uniform = awake / awake.sum(axis=1, keepdims=True)
    algo_weights = {"Hedge": W_hedge, "OGD": W_ogd, "FTRL": W_ftrl, "Uniform average": W_uniform}

    all_true = np.ones(T, dtype=bool)
    raw_ll, cal_ll = {}, {}
    for name, W in algo_weights.items():
        phat = np.einsum("tn,tnk->tk", W, P)
        raw_ll[name] = full_length_raw_logloss(phat, y, T)
        cal_ll[name] = full_length_calibrated_logloss(phat, y, all_true, T)
        print(f"  calibrated {name} done")

    for book in ["PSC", "B365C"]:
        k = panel.index(book)
        raw_ll[book] = full_length_raw_logloss(P[:, k, :], y, T, mask=awake[:, k])
        cal_ll[book] = full_length_calibrated_logloss(P[:, k, :], y, awake[:, k], T)
        print(f"  calibrated {book} done")

    comparisons = [
        ("OGD vs PSC (raw)", raw_ll["OGD"], raw_ll["PSC"]),
        ("OGD vs PSC (calibrated)", cal_ll["OGD"], cal_ll["PSC"]),
        ("OGD vs B365C (raw)", raw_ll["OGD"], raw_ll["B365C"]),
        ("OGD vs Hedge (raw)", raw_ll["OGD"], raw_ll["Hedge"]),
        ("OGD vs FTRL (raw)", raw_ll["OGD"], raw_ll["FTRL"]),
        ("OGD vs Uniform average (raw)", raw_ll["OGD"], raw_ll["Uniform average"]),
        ("Hedge vs Uniform average (raw)", raw_ll["Hedge"], raw_ll["Uniform average"]),
        ("FTRL vs Uniform average (raw)", raw_ll["FTRL"], raw_ll["Uniform average"]),
        ("Hedge: raw vs calibrated", raw_ll["Hedge"], cal_ll["Hedge"]),
        ("OGD: raw vs calibrated", raw_ll["OGD"], cal_ll["OGD"]),
        ("FTRL: raw vs calibrated", raw_ll["FTRL"], cal_ll["FTRL"]),
        ("Uniform average: raw vs calibrated", raw_ll["Uniform average"], cal_ll["Uniform average"]),
    ]

    print(f"\n=== Moving block bootstrap tests (n_boot={N_BOOT}, {CI_LEVEL:.0%} CI) ===")
    print("(negative mean_diff = first-named series has the LOWER/better log-loss)\n")
    rows = []
    for label, a, b in comparisons:
        d, n = paired_diff(a, b)
        res = moving_block_bootstrap_test(d)
        rows.append({"comparison": label, **res})
        flag = "SIGNIFICANT" if res["significant_95"] else "not significant"
        print(f"{label:38s} n={res['n']:6d}  block={res['block_length']:3d}  "
              f"mean_diff={res['mean_diff']:+.5f}  95% CI=[{res['ci_low']:+.5f}, {res['ci_high']:+.5f}]  "
              f"p={res['p_value']:.4f}  {flag}")

    table = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "significance_test_table.csv")
    table.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
