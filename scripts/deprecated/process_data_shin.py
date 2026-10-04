"""
Variant of process_data.py that de-vigs bookmaker odds with Shin's (1992,
1993) method instead of basic proportional normalization. Everything else
(raw-file discovery, bookmaker auto-detection, VC/BV merge, aggregate-column
exclusion) is IDENTICAL -- imported directly from process_data.py -- so the
only difference between odds_long.csv and odds_long_shin.csv is the de-vig
formula, which is what makes a like-for-like algorithm comparison possible
(see shin_vs_basic_comparison.py).

Shin's model: a bookmaker faces a mix of perfectly-informed "insider"
bettors (a fraction z of the money staked, who bet with certainty on the
true winner) and ordinary bettors (who bet in proportion to the true
probability). A risk-neutral bookmaker who must break even against this mix
ends up quoting raw, overround-inflated probabilities that are NOT simply
the fair probabilities scaled up by a constant factor -- unlike basic
proportional normalization (which assumes exactly that, and so spreads the
margin evenly across H/D/A), Shin's method backs out both z and the fair
probabilities jointly. This shifts probability mass from longshots toward
favorites relative to proportional normalization -- the standard correction
for the well-documented "favorite-longshot bias" in fixed-odds markets (the
same bias calibration_analysis.py independently finds in this project's
basic-normalized data).

Produces: data/processed/odds_long_shin.csv -- a parallel dataset, not read
by any of the main pipeline scripts (sleeping_experts.py etc. all hardcode
odds_long.csv). Only shin_vs_basic_comparison.py reads this file.
"""

import glob
import os
import numpy as np
import pandas as pd

from process_data import (
    RAW_DIR, OUT_DIR, LEAGUE_NAMES, AGGREGATE_PREFIXES, MERGE_ALIASES, find_odds,
)


def shin_normalize(h, d, a, n_iter=60):
    """Vectorized Shin (1992) de-vig via bisection on the insider-trading
    proportion z. h, d, a: 1-D arrays of decimal odds, one match per entry.
    Returns (pH, pD, pA, overround, z) arrays.

    Let pi_i = 1/odds_i be the raw (overround-inflated) implied
    probabilities and B = sum_j pi_j. Shin's model gives the fair
    probability as a function of z:

        p_i(z) = ( sqrt(z^2 + 4(1-z) pi_i^2 / B) - z ) / (2(1-z))

    z is the unique root in [0, 1) of sum_i p_i(z) = 1 (p_i(z) is
    monotonically decreasing in z, so bisection converges reliably). At
    z=0 this reduces exactly to proportional normalization (p_i = pi_i);
    z>0 progressively shifts mass from the longshot toward the favorite.
    overround (= B - 1) is unaffected by the de-vig method -- it depends
    only on the raw odds -- so it matches process_data.devig_normalize's
    overround column exactly."""
    pi = np.stack([1.0 / h, 1.0 / d, 1.0 / a], axis=1)   # (n, 3)
    B = pi.sum(axis=1, keepdims=True)                     # (n, 1)

    def p_of_z(z):
        return (np.sqrt(z**2 + 4.0 * (1.0 - z) * pi**2 / B) - z) / (2.0 * (1.0 - z))

    lo = np.zeros((len(h), 1))
    hi = np.full((len(h), 1), 0.4)  # z is small in practice (a few %); 0.4 is a generous bracket
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        z_too_large = p_of_z(mid).sum(axis=1, keepdims=True) < 1.0  # sum(p) decreasing in z
        hi = np.where(z_too_large, mid, hi)
        lo = np.where(z_too_large, lo, mid)

    z = (lo + hi) / 2.0
    p = p_of_z(z)
    p = p / p.sum(axis=1, keepdims=True)  # guard against residual bisection imprecision
    overround = B.flatten() - 1.0
    return p[:, 0], p[:, 1], p[:, 2], overround, z.flatten()


def load_one_file(path):
    """Same per-file processing as process_data.load_one_file(), except the
    de-vig step is shin_normalize() instead of devig_normalize() -- see that
    function's docstring for what's shared vs. different."""
    code_season = os.path.basename(path).replace(".csv", "")
    div_code, season = code_season.split("_")
    df = pd.read_csv(path, encoding="latin1")

    df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    df["League"] = div_code
    df["LeagueName"] = LEAGUE_NAMES.get(div_code, div_code)
    df["Season"] = season
    df["MatchID"] = code_season + "_" + df.index.astype(str)

    triples = find_odds(df.columns)
    expert_prefixes = sorted(p for p in triples if p not in AGGREGATE_PREFIXES)

    base_cols = ["MatchID", "League", "LeagueName", "Season", "Date", "HomeTeam", "AwayTeam", "FTR"]
    meta = df[base_cols].copy()

    long_rows = []
    for prefix in expert_prefixes:
        canonical = MERGE_ALIASES.get(prefix, prefix)
        colH, colD, colA = triples[prefix]
        h = pd.to_numeric(df[colH], errors="coerce")
        d = pd.to_numeric(df[colD], errors="coerce")
        a = pd.to_numeric(df[colA], errors="coerce")
        valid = h.notna() & d.notna() & a.notna() & (h > 1) & (d > 1) & (a > 1)
        if not valid.any():
            continue

        pH, pD, pA, overround, z = shin_normalize(
            h[valid].to_numpy(), d[valid].to_numpy(), a[valid].to_numpy()
        )

        sub = df.loc[valid, ["MatchID", "League", "LeagueName", "Season", "Date", "FTR"]].copy()
        sub["Bookmaker"] = canonical
        sub["pH"], sub["pD"], sub["pA"] = pH, pD, pA
        sub["overround"] = overround
        sub["shin_z"] = z
        sub["odds_H"], sub["odds_D"], sub["odds_A"] = h[valid].to_numpy(), d[valid].to_numpy(), a[valid].to_numpy()
        long_rows.append(sub)

    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    return meta, long_df


def main():
    """Runs load_one_file() over every raw CSV in data/raw/ (same file set
    as process_data.py) and saves odds_long_shin.csv. Prints only what
    differs from process_data.py's report -- the per-bookmaker mean insider-
    trading proportion z -- since total matches / league mix / overround are
    identical to the basic-normalized run by construction."""
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    print(f"Found {len(files)} raw files.")

    meta_frames, long_frames = [], []
    for path in files:
        meta, long_df = load_one_file(path)
        if meta is not None:
            meta_frames.append(meta)
        if long_df is not None and not long_df.empty:
            long_frames.append(long_df)

    matches_meta = pd.concat(meta_frames, ignore_index=True)
    odds_long = pd.concat(long_frames, ignore_index=True)

    out_path = os.path.join(OUT_DIR, "odds_long_shin.csv")
    odds_long.to_csv(out_path, index=False)

    print("\n=== OVERALL ===")
    print(f"Total matches: {len(matches_meta)}  (identical universe to odds_long.csv)")

    print("\n=== MEAN SHIN INSIDER-TRADING PROPORTION (z, %) BY BOOKMAKER ===")
    print((odds_long.groupby("Bookmaker")["shin_z"].mean() * 100).round(2).sort_values())

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
