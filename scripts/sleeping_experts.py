"""
Extension of an earlier 6-bookmaker first draft (not carried into this
project) to the FULL bookmaker universe, including the
"softer" bookmakers that only cover a subset of matches/seasons (see the
coverage report printed by process_data.py).

Because not every bookmaker quotes odds for every match, the standard Hedge /
OGD / FTRL algorithms don't directly apply (they assume every expert reports a
prediction every round). We use the classic SLEEPING EXPERTS / SPECIALISTS
reduction (Freund, Schapire, Singer & Warmuth, 1997; see also Blum & Mansour,
2007) for all three algorithms:

  - each expert keeps its own persistent state (weight for Hedge, position for
    OGD, cumulative loss for FTRL);
  - at round t only the "awake" experts A(t) (the ones quoting odds for that
    match) are used to form the prediction, renormalized to a distribution
    over A(t) alone;
  - only awake experts' state is updated after round t; asleep experts are
    frozen exactly as they were;
  - a round where NOBODY is awake is skipped entirely (its W row stays all
    zeros and no state moves). load_full_universe() guarantees >=1 quote per
    round on the full panel, and the subset builders elsewhere filter to
    >=1 awake, so this never fires in the standard pipeline -- but it does
    as soon as a column is dropped from the panel (e.g. leaving one
    bookmaker out when it was the only quote for some match), which used to
    raise IndexError out of project_to_simplex.

This gives each expert i a per-expert regret guarantee measured only over the
rounds it was actually awake, rather than requiring it to have participated in
every round.

ALGORITHMS / DATA LOADING ONLY -- this is the shared library almost every
other script in the project imports from. The actual experiment that runs
these algorithms over the full panel and builds results_table.csv +
sleeping_experts_comparison.png lives in sleeping_experts_experiment.py.
"""

import hashlib
import os
import numpy as np
import pandas as pd

# hardcoded relative to scripts/ -- scripts always run from inside scripts/
# (see CLAUDE.md), data/ and results/ just need to be its sibling folders
DATA_DIR = os.path.join("..", "data", "processed")
OUT_DIR = os.path.join("..", "results")
CACHE_DIR = os.path.join("..", "data", "cache")

EPS = 1e-6                      # clip probabilities away from 0 so log-loss stays finite
M = np.log(1.0 / EPS)           # resulting bound on any single round's per-expert loss
OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}

# OGD's step size is scaled up by this factor -- see learning_rate_study.py.
# Why it is needed: M = log(1/EPS) = 13.82 bounds a round's per-expert loss
# assuming a bookmaker might price the realised outcome at 1e-6, whereas real
# de-vigged football probabilities bottom out near 2% (observed max loss:
# 3.76). eta ~ 1/M is therefore far more conservative than the data warrants.
# Selected the same way this project selects the calibration learning rate
# (calibration_correction.select_lr_on_train): a grid, evaluated on a
# chronological prefix only. Train and held-out tail BOTH put the optimum at
# 100 (train 0.99739, held-out 0.99765, against 0.99826/0.99803 at 1.0), and
# it is an interior optimum -- 200, 500 and 1000 are all worse.
#
# The direction is specific to OGD: Hedge and FTRL want their step scaled DOWN
# instead (see below), so all three are tuned, on the same protocol, and the
# OGD-vs-Hedge/FTRL comparison stays fair. The asymmetry is structural -- OGD's
# eta multiplies one round's gradient, while FTRL's multiplies the whole
# accumulated loss, so scaling FTRL up collapses it toward follow-the-leader.
#
# Principled alternatives were tried and recover only a fraction of this:
# an honest a-priori bound M=log(100) gives 15%, AdaGrad (scalar) 20%, and
# per-expert AdaGrad is worse than no scaling at all.
OGD_STEP_MULTIPLIER = 100.0

# Hedge and FTRL want the step scaled DOWN, not up. The original sweep reported
# both as already optimal at 1.0, but its grid STARTED at 1.0 -- so that was a
# boundary, not an optimum, and nobody had looked to its left. Re-running the
# search downwards finds a genuine interior minimum for each, selected on the
# same chronological train prefix used everywhere else:
#   Hedge  train 0.998691 (x0.001) -> 0.998118 (x0.25) -> 0.998128 (x1)
#   FTRL   train 0.998661 (x0.001) -> 0.998117 (x0.05) -> 0.998160 (x1)
# Held-out gain over the old x1: +0.00013 for Hedge, +0.00018 for FTRL. The
# direction is the mirror image of OGD's and for the same structural reason --
# Hedge's eta sits in an exponent over a single round's loss, so shrinking it
# lengthens the memory, while OGD's multiplies one gradient step.
HEDGE_STEP_MULTIPLIER = 0.25
FTRL_STEP_MULTIPLIER = 0.05

COLORS = {
    "Hedge": "#2a78d6",
    "OGD": "#eb6834",
    "FTRL": "#1baf7a",
    "Uniform average": "#4a3aa7",
    "benchmark": "#52514e",
    "muted": "#898781",
}


# ---------------------------------------------------------------- data -----

EXCLUDED_BOOKMAKERS = ()
"""Empty ON PURPOSE -- nothing is excluded by name. Kept as a hook (the four
panel builders still apply it) but deliberately not used.

HISTORY, because this was reversed once and the reasoning matters. Interwetten
(IW, IWC) used to be listed here: it stops quoting on 2024-01-04 and never
returns, yet over 10 seasons still shows 74.7% coverage and cleared the Tier-A
floor, which truncated the "rounds where every compared series is defined"
intersection at its last quote and pushed the held-out tail into the MIDDLE of
the data.

That was a real bug, but excluding IW by name was the wrong fix, for two
reasons:

  1. **It was not symmetric.** The panel keeps 13 series with 0% coverage in
     the first eight seasons (BFE/BFEC at 19%, 1XB/BF/BFC/1XBC at ~10%,
     BFD/BFDC/BMGM/BMGMC/LBC/CL/CLC at 7-10%), i.e. the exact mirror image of
     IW's pathology. IW is the SEVENTH best-covered series of the 26 and was
     the only one removed. If "dead at the end" disqualifies, "absent at the
     start" must too, and the panel collapses to about six series.
  2. **It treated the symptom.** The defect was in the comparison
     INSTRUMENTS -- a coverage floor computed over the whole 10 seasons, and a
     global intersection across every compared series. Those have since been
     fixed where they mattered: recent_window.py recomputes its coverage floor
     INSIDE its own window (so IW fails it there on its own merits, at 0%), and
     best_configuration.py intersects only over the series it actually
     compares, which never included IW.

Keeping IW is also the better science on its own terms: the Sleeping Experts
reduction exists precisely to handle an expert that is absent for part of the
history, and the per-expert regret guarantee is measured over each expert's own
awake rounds. An expert that falls permanently asleep is the motivating case
for the machinery, not an exception to it -- and it is the sharpest example in
this dataset of the non-stationarity that motivates fixed_share.py and explains
why the Aggregating Algorithm underperforms here."""


def _build_universe(csv_path):
    """The panel construction itself. Returns the usual six values plus the
    MatchID index, which load_match_order() needs and which costs nothing to
    carry along here."""
    long = pd.read_csv(csv_path, parse_dates=["Date"])
    long = long[~long["Bookmaker"].isin(EXCLUDED_BOOKMAKERS)]
    panel = sorted(long["Bookmaker"].unique().tolist())
    N = len(panel)

    pivot = long.pivot(index="MatchID", columns="Bookmaker", values=["pH", "pD", "pA"])
    meta_all = long.drop_duplicates("MatchID").set_index("MatchID")[["Date", "FTR", "Season"]]

    awake_any = pivot["pH"].notna().any(axis=1)  # keep matches with >=1 quoting bookmaker
    match_ids = pivot.index[awake_any]
    pivot = pivot.loc[match_ids]
    meta = meta_all.loc[match_ids].sort_values("Date", kind="mergesort")
    pivot = pivot.loc[meta.index]

    T = len(meta)
    P = np.full((T, N, 3), np.nan)
    for k, book in enumerate(panel):
        P[:, k, 0] = pivot[("pH", book)].values
        P[:, k, 1] = pivot[("pD", book)].values
        P[:, k, 2] = pivot[("pA", book)].values

    awake = ~np.isnan(P[:, :, 0])
    P = np.clip(np.nan_to_num(P, nan=EPS), EPS, 1.0)

    y = meta["FTR"].map(OUTCOME_INDEX).values
    dates = meta["Date"].values
    seasons = meta["Season"].astype(str).values
    return panel, P, awake, y, dates, seasons, meta.index.values


def _cached_universe(csv_path):
    """_build_universe with an on-disk cache.

    Parsing a 123 MB CSV costs about 2.4s and the pipeline does it some fifty
    times, so the result is memoised. The key is the source file's path, size
    and modification time, which means regenerating or editing odds_long.csv
    invalidates the cache automatically: there is no way to read a stale panel,
    and no cache-clearing step to forget."""
    st = os.stat(csv_path)
    key = (f"{os.path.abspath(csv_path)}|{st.st_size}|{st.st_mtime_ns}"
           f"|{sorted(EXCLUDED_BOOKMAKERS)}")
    cp = os.path.join(CACHE_DIR, hashlib.sha256(key.encode()).hexdigest()[:16] + ".npz")

    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=False)
        # seasons and match_id are stored as fixed-width unicode because npz
        # refuses to load object arrays without pickle; they are handed back as
        # object arrays so callers see exactly what the uncached path returns
        return (list(z["panel"]), z["P"], z["awake"], z["y"], z["dates"],
                z["seasons"].astype(object), z["match_id"].astype(object))

    built = _build_universe(csv_path)
    os.makedirs(CACHE_DIR, exist_ok=True)
    # written through a handle and renamed into place, so a cache file only ever
    # exists complete: np.savez would otherwise append .npz to the temp name,
    # and a crash mid-write would leave a partial file behind
    tmp = cp + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, panel=np.array(built[0]), P=built[1], awake=built[2],
                 y=built[3], dates=built[4],
                 seasons=np.asarray(built[5], dtype=str),
                 match_id=np.asarray(built[6], dtype=str))
    os.replace(tmp, cp)
    return built


def load_universe_from(csv_path):
    """Build a (rounds x experts x outcomes) tensor plus a boolean 'awake' mask,
    sorted chronologically, from any long-format odds CSV. Used to read either
    the basic- or the Shin-normalized dataset with identical logic.
    EXCLUDED_BOOKMAKERS never enter the panel."""
    return _cached_universe(csv_path)[:6]


def load_full_universe():
    """load_universe_from() on the project's default (basic-normalized) data."""
    return load_universe_from(os.path.join(DATA_DIR, "odds_long.csv"))


def load_match_order():
    """The MatchIDs in the same order as round t = 0..T-1, so a bookmaker's raw
    decimal odds (which the tensor does not keep, it holds de-vig'd
    probabilities) can be aligned to the same round index as everything else.
    The ordering is normalization-independent, since de-vig cannot change which
    matches have usable odds, so the basic file is always the one read."""
    return _cached_universe(os.path.join(DATA_DIR, "odds_long.csv"))[6]


def split_by_market_phase(panel):
    """A processed bookmaker name is the CLOSING-odds counterpart of the
    identically-prefixed OPENING one iff it ends in "C" -- football-data.co.uk's
    own column convention, which data_processer.py preserves verbatim. Returns
    (open_idx, close_idx), the panel indices of each subset."""
    close_idx = [k for k, name in enumerate(panel) if name.endswith("C")]
    open_idx = [k for k, name in enumerate(panel) if not name.endswith("C")]
    unpaired = [panel[k] for k in close_idx if panel[k][:-1] not in panel]
    if unpaired:
        print(f"  NOTE: closing-odds series with no matching opening name: {unpaired}")
    return open_idx, close_idx


def project_to_simplex(v):
    """Euclidean projection onto the probability simplex (Duchi et al., 2008)."""
    n = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    idx = np.arange(1, n + 1)
    cond = u - css / idx > 0
    rho = idx[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def per_round_expert_loss(P, y):
    """l[t, i] = -log(P[t, i, y[t]]) (only meaningful where awake[t, i] is True)."""
    T = P.shape[0]
    q = P[np.arange(T), :, y]
    return -np.log(q)


# ------------------------------------------------- sleeping-experts algos --

def run_hedge_sleeping(loss, awake, N):
    """Sleeping-experts Hedge (see module docstring). NOTE: `w` is the raw,
    persistent per-expert state and is NEVER globally renormalized to sum to
    1 -- only the awake-restricted copy `v` is normalized, and only for the
    purpose of making a prediction. This detail matters if you extend this
    function: see fixed_share.py's comment about a bug this caused there."""
    w = np.full(N, 1.0 / N)               # persistent weight, frozen while asleep
    T = loss.shape[0]
    W = np.zeros((T, N))
    for t in range(T):
        a = awake[t]
        if not a.any():
            continue                      # nobody quoting: nothing to predict, nothing to learn
        v = w[a] / w[a].sum()             # renormalize over the awake subset only
        W[t, a] = v
        eta = HEDGE_STEP_MULTIPLIER * np.sqrt(np.log(N) / (t + 1)) / M
        w[a] = w[a] * np.exp(-eta * loss[t, a])
    return W


def run_ogd_sleeping(loss, awake, N):
    """Sleeping-experts Online Gradient Descent: project-then-step, where the
    projection is re-done onto whatever subset is awake THIS round (a no-op
    if that subset's weights are already feasible from last time it was
    used) -- see module docstring for why this exactly reduces to plain OGD
    when every expert is always awake.

    The step size carries OGD_STEP_MULTIPLIER (see its comment above): the
    theory-optimal schedule is far too conservative on this data, and the
    multiplier was selected on a chronological train prefix. Set it to 1.0 to
    recover the pre-tuning behaviour."""
    D = np.sqrt(2.0)
    w = np.full(N, 1.0 / N)               # persistent, always-feasible-when-last-touched position
    T = loss.shape[0]
    W = np.zeros((T, N))
    for t in range(T):
        a = awake[t]
        if not a.any():
            continue                      # nobody quoting: nothing to predict, nothing to learn
        v = project_to_simplex(w[a])      # re-feasible over today's awake subset (no-op if unchanged)
        W[t, a] = v
        w[a] = v
        eta = OGD_STEP_MULTIPLIER * D / (M * np.sqrt(t + 1))
        w[a] = w[a] - eta * loss[t, a]    # greedy gradient step, only on awake coordinates
    return W


def run_ftrl_sleeping(loss, awake, N):
    """Sleeping-experts FTRL (L2 regularizer): re-derives each round's weights
    from the ENTIRE cumulative loss so far (only incremented on awake
    rounds -- frozen while asleep), rather than stepping from last round's
    weights like OGD does. "Lazy" vs. "greedy", see module docstring."""
    D = np.sqrt(2.0)
    w0 = np.full(N, 1.0 / N)
    cum_loss = np.zeros(N)                # frozen (not incremented) while asleep
    T = loss.shape[0]
    W = np.zeros((T, N))
    for t in range(T):
        a = awake[t]
        if not a.any():
            continue                      # nobody quoting: nothing to predict, nothing to learn
        eta = FTRL_STEP_MULTIPLIER * D / (M * np.sqrt(t + 1))
        v = project_to_simplex(w0[a] - eta * cum_loss[a])
        W[t, a] = v
        cum_loss[a] += loss[t, a]
    return W


# --------------------------------------------------------- weight readout --

def last_awake_weight(W, awake):
    """Each expert's mixture weight at the LAST round it was actually awake.

    Why not just read W[-1]: W[t, i] is only written for experts awake at
    round t (see the three run_*_sleeping functions above -- the asleep
    coordinates of that row stay 0), so the final row is exactly 0 for every
    expert that didn't quote the LAST match of the dataset. On the current
    panel that's 10 of the 26 bookmakers, PSC (the single best predictor
    overall) among them. Read as a "final weight", that 0 means "didn't
    quote the last match", NOT "the algorithm learned to ignore this
    expert" -- and it silently hides exactly the frozen-advantage cases
    fixed_share.py exists to diagnose (a single-season bookmaker whose
    season ended before the end of the dataset keeps a large weight, yet
    would report 0.0).

    Returns an (N,) array, NaN for an expert that was never awake. Each
    entry is a share of the awake set AT ITS OWN round, so entries are
    individually interpretable ("how much of that round's prediction was
    this expert responsible for, the last time it took part") but do NOT sum
    to 1 across experts."""
    ever = awake.any(axis=0)
    last_idx = awake.shape[0] - 1 - np.argmax(awake[::-1], axis=0)
    out = np.full(awake.shape[1], np.nan)
    cols = np.flatnonzero(ever)
    out[cols] = W[last_idx[cols], cols]
    return out


# ------------------------------------------------------------- metrics -----

def evaluate(probs, y, mask=None):
    """log-loss, Brier, RPS (order H,D,A) and accuracy, averaged over masked rounds."""
    T = len(y)
    if mask is None:
        mask = np.ones(T, dtype=bool)
    p = probs[mask]
    yy = y[mask]
    idx = np.arange(len(yy))

    q = np.clip(p[idx, yy], EPS, 1.0)
    log_loss = -np.log(q)

    onehot = np.zeros_like(p)
    onehot[idx, yy] = 1.0
    brier = ((p - onehot) ** 2).sum(axis=1)

    cumP1, cumP2 = p[:, 0], p[:, 0] + p[:, 1]
    cumO1, cumO2 = onehot[:, 0], onehot[:, 0] + onehot[:, 1]
    rps = 0.5 * ((cumP1 - cumO1) ** 2 + (cumP2 - cumO2) ** 2)

    acc = (p.argmax(axis=1) == yy).astype(float)

    return {
        "rounds": int(mask.sum()),
        "log_loss": log_loss.mean(),
        "brier": brier.mean(),
        "rps": rps.mean(),
        "accuracy": acc.mean(),
    }
