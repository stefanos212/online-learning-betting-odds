"""
Load raw football-data.co.uk CSVs,detects bookmaker 1X2 odds triples,
de-vig them into clean probabilities, and produce the processed dataset
"""

import re
import glob
import os
import pandas as pd

# hardcoded relative to scripts/ -- scripts always run from inside scripts/
# (see CLAUDE.md), data/ just needs to be its sibling folder in the project
RAW_DIR = os.path.join("..", "data", "raw")
OUT_DIR = os.path.join("..", "data", "processed")

LEAGUE_NAMES = {
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

# aggregate columns that are themselves combinations of other bookmakers,
# not an independent "expert" -> exclude from the expert panel, keep as reference only
# BbAv/BbMx (Betbrain average/max) only appear in older seasons (pre ~2020/21,
# before football-data.co.uk switched to the Max/Avg naming) -- absent from
# the original 2021+-only dataset, which is why this wasn't caught earlier.
AGGREGATE_PREFIXES = {"Max", "Avg", "MaxC", "AvgC", "BbAv", "BbMx", "BbAvC", "BbMxC"}

# VC Bet rebranded to BetVictor (BV) partway through our seasons (VC present
# 2021/22-2023/24, BV present from 2025/26 on, never both in the same season)
# -- same underlying odds-setting entity, so merge them into one continuous
# expert identity instead of two experts with a mysterious clean handover.
MERGE_ALIASES = {"VC": "VC_BV", "BV": "VC_BV", "VCC": "VC_BVC", "BVC": "VC_BVC"}

TRIPLE_RE = re.compile(r"^(.+)H$")


def find_odds(columns):
    """Return dict prefix -> (colH, colD, colA) for every H/D/A odds triple in the columns."""
    triples = {}
    cols = set(columns)
    for col in columns:
        m = TRIPLE_RE.match(col)
        if not m:
            continue
        prefix = m.group(1)
        colD, colA = prefix + "D", prefix + "A"
        if colD in cols and colA in cols:
            triples[prefix] = (col, colD, colA)
    return triples


def devig_normalize(h, d, a):
    """Vectorized basic-normalization de-vig: p_i = (1/odds_i) / sum(1/odds_j).
    h, d, a are arrays of decimal odds; returns (pH, pD, pA, overround) arrays."""
    inv_h, inv_d, inv_a = 1.0 / h, 1.0 / d, 1.0 / a
    total = inv_h + inv_d + inv_a
    overround = total - 1.0
    return inv_h / total, inv_d / total, inv_a / total, overround


def load_one_file(path):
    """Process one raw "<LeagueCode>_<Season>.csv" file into:
      - meta: base per-match columns (League/Season/Date/FTR/...), used only
        for main()'s exploration report -- not saved to disk
      - long_df: one row per (match, bookmaker) that actually quoted usable
        odds for that match (used to build odds_long.csv, the format every
        downstream online-learning script actually reads)"""
    code_season = os.path.basename(path).replace(".csv", "")
    div_code, season = code_season.split("_")
    df = pd.read_csv(path, encoding="latin1")

    df = df.dropna(subset=["FTR", "HomeTeam", "AwayTeam"]).copy()
    # format="mixed": older seasons use dd/mm/yy, newer ones dd/mm/yyyy -- pandas
    # can't infer one consistent format across them, "mixed" parses row-by-row
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
        # coerce to numeric: some of the newly-added leagues have occasional
        # stray non-numeric values in odds columns (e.g. a misformatted row),
        # which would otherwise crash the "> 1" comparison below -- turning
        # them into NaN lets the existing notna() check correctly drop just
        # that one match/bookmaker instead of the whole file failing
        h = pd.to_numeric(df[colH], errors="coerce")
        d = pd.to_numeric(df[colD], errors="coerce")
        a = pd.to_numeric(df[colA], errors="coerce")
        valid = h.notna() & d.notna() & a.notna() & (h > 1) & (d > 1) & (a > 1)
        if not valid.any():
            continue

        pH, pD, pA, overround = devig_normalize(
            h[valid].to_numpy(), d[valid].to_numpy(), a[valid].to_numpy()
        )

        sub = df.loc[valid, ["MatchID", "League", "LeagueName", "Season", "Date", "FTR"]].copy()
        sub["Bookmaker"] = canonical
        sub["pH"], sub["pD"], sub["pA"] = pH, pD, pA
        sub["overround"] = overround
        sub["odds_H"], sub["odds_D"], sub["odds_A"] = h[valid].to_numpy(), d[valid].to_numpy(), a[valid].to_numpy()
        long_rows.append(sub)

    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    return meta, long_df


def main():
    """Runs load_one_file() over every raw CSV in data/raw/, concatenates the
    long-format rows into odds_long.csv (the one processed dataset every
    downstream script reads), and prints a coverage/exploration report used
    to sanity-check the data -- computed from an in-memory, unsaved per-match
    frame (which is how the single-season and VC/BV-rebrand issues were
    first noticed -- see sleeping_experts.py's single_season warning and the
    MERGE_ALIASES comment above)."""
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

    odds_long.to_csv(os.path.join(OUT_DIR, "odds_long.csv"), index=False)

    # ---- exploration report ----
    print("\n=== OVERALL ===")
    print(f"Total matches: {len(matches_meta)}")
    print(f"Leagues x seasons: {matches_meta.groupby(['League','Season']).size().shape[0]} combos")
    print(f"Date range: {matches_meta['Date'].min()} -> {matches_meta['Date'].max()}")

    print("\n=== MATCHES PER LEAGUE ===")
    print(matches_meta.groupby("LeagueName").size().sort_values(ascending=False))

    print("\n=== RESULT DISTRIBUTION (FTR) ===")
    print(matches_meta["FTR"].value_counts(normalize=True).round(3))

    print("\n=== BOOKMAKERS FOUND & COVERAGE (% of matches with usable odds) ===")
    cov = odds_long.groupby("Bookmaker")["MatchID"].nunique().sort_values(ascending=False)
    total = matches_meta["MatchID"].nunique()
    cov_pct = (cov / total * 100).round(1)
    report = pd.DataFrame({"matches_covered": cov, "coverage_pct": cov_pct})
    print(report)

    print("\n=== BOOKMAKER OVERROUND (avg market margin, %) ===")
    print((odds_long.groupby("Bookmaker")["overround"].mean() * 100).round(2).sort_values())

    # candidate "core panel": bookmakers present across (almost) all seasons/leagues
    per_bookmaker_season_leagues = odds_long.groupby("Bookmaker").apply(
        lambda g: g[["League", "Season"]].drop_duplicates().shape[0]
    )
    n_combos = matches_meta.groupby(["League", "Season"]).ngroups
    print(f"\n=== BOOKMAKERS PRESENT IN ALL {n_combos} (league,season) COMBOS ===")
    full_coverage = per_bookmaker_season_leagues[per_bookmaker_season_leagues == n_combos]
    print(sorted(full_coverage.index.tolist()))

    print(f"\nSaved: {os.path.join(OUT_DIR, 'odds_long.csv')}")


if __name__ == "__main__":
    main()
