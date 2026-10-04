import glob
import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join("..", "data", "raw")
OUT_DIR = os.path.join("..", "data", "processed")

LEAGUES = {
    "E0": "England_PremierLeague", "E1": "England_Championship",
    "E2": "England_League1", "E3": "England_League2", "EC": "England_Conference",
    "SC0": "Scotland_Premiership", "SC1": "Scotland_Championship",
    "SC2": "Scotland_League1", "SC3": "Scotland_League2",
    "I1": "Italy_SerieA", "I2": "Italy_SerieB",
    "SP1": "Spain_LaLiga", "SP2": "Spain_SegundaDivision",
    "D1": "Germany_Bundesliga", "D2": "Germany_2Bundesliga",
    "F1": "France_Ligue1", "F2": "France_Ligue2",
    "N1": "Netherlands_Eredivisie", "B1": "Belgium_JupilerLeague",
    "P1": "Portugal_LigaI", "T1": "Turkey_SuperLig", "G1": "Greece_SuperLeague",
}

AGGREGATE_PREFIXES = {"Max", "Avg", "MaxC", "AvgC", "BbAv", "BbMx", "BbAvC", "BbMxC"}

MERGE_ALIASES = {"VC": "VC_BV", "BV": "VC_BV", "VCC": "VC_BVC", "BVC": "VC_BVC"}

KNOWN_BOOKMAKER_PREFIXES = {
    "PSC", "BFC", "BFEC", "1XBC", "LB", "LBC", "CLC", "B365C", "BMGMC",
    "VC", "BV", "VCC", "BVC",   # VC/BV rebrand: ποτέ και τα δύο στην ίδια σεζόν
    "BWC", "WHC", "BF", "PS", "BFDC", "IWC", "B365", "WH", "BW", "IW",
    "BMGM", "1XB", "BFD", "CL", "BFE",
}


def find_bookmakers_odds(columns):
    """prefix -> (colH, colD, colA) for every known bookmaker whose full
    H/D/A triple is present in `columns`."""
    cols = set(columns)
    triples = {}
    for prefix in KNOWN_BOOKMAKER_PREFIXES:
        colH, colD, colA = prefix + "H", prefix + "D", prefix + "A"
        if colH in cols and colD in cols and colA in cols:
            triples[prefix] = (colH, colD, colA)
    return triples


def devig_normalize(h, d, a):
    """Proportional normalization: p_i = (1/odds_i) / sum(1/odds_j)."""
    inv_h, inv_d, inv_a = 1.0 / h, 1.0 / d, 1.0 / a
    total = inv_h + inv_d + inv_a
    overround = total - 1.0
    return inv_h / total, inv_d / total, inv_a / total, overround


def shin_normalize(h, d, a, n_iter=60):
    """Shin (1992) de-vig, one match at a time (a simple loop -- slower
    than a vectorized version, but easier to follow).

    For each match, we need a number z (the "insider-trading proportion")
    such that the three fair probabilities
        p_i(z) = ( sqrt(z^2 + 4(1-z) pi_i^2 / B) - z ) / (2(1-z))
    (where pi_i = 1/odds_i and B = pi_H + pi_D + pi_A) add up to 1. There's
    no formula to solve for z directly, so we just try values of z with
    binary search: start with a wide range [0, 0.4], and keep halving it,
    each time checking whether the current guess gives a sum that's too
    big or too small, until the range is tiny."""
    n = len(h)
    pH = np.zeros(n)
    pD = np.zeros(n)
    pA = np.zeros(n)
    overround = np.zeros(n)
    z_values = np.zeros(n)

    for i in range(n):
        pi_h = 1.0 / h[i]
        pi_d = 1.0 / d[i]
        pi_a = 1.0 / a[i]
        B = pi_h + pi_d + pi_a

        lo = 0.0
        hi = 0.4  # z is small in practice (a few %), so this is a generous starting range
        for _ in range(n_iter):
            z = (lo + hi) / 2.0
            p_h = ((z**2 + 4 * (1 - z) * pi_h**2 / B) ** 0.5 - z) / (2 * (1 - z))
            p_d = ((z**2 + 4 * (1 - z) * pi_d**2 / B) ** 0.5 - z) / (2 * (1 - z))
            p_a = ((z**2 + 4 * (1 - z) * pi_a**2 / B) ** 0.5 - z) / (2 * (1 - z))
            if p_h + p_d + p_a < 1.0:
                hi = z  # z was too large, search the lower half next
            else:
                lo = z  # z was too small, search the upper half next

        total = p_h + p_d + p_a  # should already be ~1, this just cleans up rounding
        pH[i] = p_h / total
        pD[i] = p_d / total
        pA[i] = p_a / total
        overround[i] = B - 1.0
        z_values[i] = z

    return pH, pD, pA, overround, z_values


def load_one_file(path):
    """One raw '<League>_<Season>.csv' -> two long-format odds frames
    (basic proportional, Shin), computed from the same raw odds so only
    the de-vig formula differs between them."""
    code_season = os.path.basename(path).replace(".csv", "")
    div_code, season = code_season.split("_")
    df = pd.read_csv(path, encoding="latin1")

    df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    df["League"] = div_code
    df["LeagueName"] = LEAGUES.get(div_code, div_code)
    df["Season"] = season
    df["MatchID"] = code_season + "_" + df.index.astype(str)

    triples = find_bookmakers_odds(df.columns)
    expert_prefixes = sorted(p for p in triples if p not in AGGREGATE_PREFIXES)

    basic_rows, shin_rows = [], []
    for prefix in expert_prefixes:
        canonical = MERGE_ALIASES.get(prefix, prefix)
        colH, colD, colA = triples[prefix]
        h = pd.to_numeric(df[colH], errors="coerce")
        d = pd.to_numeric(df[colD], errors="coerce")
        a = pd.to_numeric(df[colA], errors="coerce")
        valid = h.notna() & d.notna() & a.notna() & (h > 1) & (d > 1) & (a > 1)
        if not valid.any():
            continue

        hv, dv, av = h[valid].to_numpy(), d[valid].to_numpy(), a[valid].to_numpy()
        base = df.loc[valid, ["MatchID", "League", "LeagueName", "Season", "Date", "FTR"]].copy()
        base["Bookmaker"] = canonical

        pH, pD, pA, overround = devig_normalize(hv, dv, av)
        basic = base.copy()
        basic["pH"], basic["pD"], basic["pA"] = pH, pD, pA
        basic["overround"] = overround
        basic["odds_H"], basic["odds_D"], basic["odds_A"] = hv, dv, av
        basic_rows.append(basic)

        pH, pD, pA, overround, z = shin_normalize(hv, dv, av)
        shin = base.copy()
        shin["pH"], shin["pD"], shin["pA"] = pH, pD, pA
        shin["overround"] = overround
        shin["shin_z"] = z
        shin["odds_H"], shin["odds_D"], shin["odds_A"] = hv, dv, av
        shin_rows.append(shin)

    basic_df = pd.concat(basic_rows, ignore_index=True) if basic_rows else pd.DataFrame()
    shin_df = pd.concat(shin_rows, ignore_index=True) if shin_rows else pd.DataFrame()
    return basic_df, shin_df


def main():
    """Builds both de-vig'd datasets from every raw file and saves them --
    nothing else (no report)."""
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))

    basic_frames, shin_frames = [], []
    for path in files:
        basic_df, shin_df = load_one_file(path)
        if not basic_df.empty:
            basic_frames.append(basic_df)
        if not shin_df.empty:
            shin_frames.append(shin_df)

    pd.concat(basic_frames, ignore_index=True).to_csv(os.path.join(OUT_DIR, "odds_long.csv"), index=False)
    pd.concat(shin_frames, ignore_index=True).to_csv(os.path.join(OUT_DIR, "odds_long_shin.csv"), index=False)


if __name__ == "__main__":
    main()
