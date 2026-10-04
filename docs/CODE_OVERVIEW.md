# Code overview

Index of every file in `scripts/`: what it does, what it depends on, every
function it defines, and what it produces in `results/` or `docs/`. Written
because the project grew across many sessions and it's easy to lose track of
which file does what and which files depend on which.

## Pipeline order / dependency graph

```
download_data.ps1
        |
        v
data_processer.py  <-- produces BOTH processed datasets in one run
        |
        +--> data/processed/odds_long.csv
        |         |
        |         v
        |    sleeping_experts.py  <-- algorithms/data-loading library only, no main();
        |         |                   everything below imports from it, directly or not
        |         +--> sleeping_experts_experiment.py  (produces results_table.csv,
        |         |                                      which several files below read)
        |         |
        |         +--> calibration_analysis.py
        |         |         |
        |         |         v
        |         +--> calibration_correction.py
        |         |         |
        |         |         v
        |         +--> final_ranking.py  <-- calibrate_full_history() is imported by the 6 files below
        |         |         |
        |         |         +--> significance_test.py   (bootstrap engine, reused by 4 more files)
        |         |         +--> value_betting.py                (reads BOTH odds_long.csv and
        |         |         |         |                            odds_long_shin.csv internally)
        |         |         |         +--> value_betting_online_kelly.py
        |         |         |         +--> value_betting_ogd_vs_ftrl.py  (also reads
        |         |         |                value_betting_all_bookmakers_table.csv, to verify
        |         |         |                its OGD half reproduces it)
        |         |         +--> market_devig_comparisons.py     (reads BOTH datasets internally too;
        |         |         |                                      reads results/results_table.csv)
        |         |         +--> contextual_experts.py
        |         |         +--> pooling_window_comparison.py
        |         |         +--> calibration_per_season.py
        |         |                   |
        |         |                   v
        |         |              calibration_warmstart.py
        |         |
        |         +--> fixed_share.py             (only depends on sleeping_experts.py)
        |         +--> outcome_class_accuracy.py  (only depends on sleeping_experts.py)
        |         +--> temporal_trends.py         (+ calibration_analysis.py helpers)
        |
        +--> data/processed/odds_long_shin.csv   (read directly by value_betting.py and
                                                     market_devig_comparisons.py, above)
```

`scripts/deprecated/`: `process_data.py` + `process_data_shin.py` (the
original, separate two-script version of `data_processer.py`), and the 8
scripts consolidated into `value_betting.py` / `market_devig_comparisons.py`
(see those two sections below, and `scripts/deprecated/README.md`) --
superseded, kept for reference only, nothing imports from them.

The write-up lives in `docs/thesis_report.tex` (Greek, XeLaTeX); it embeds several
`results/*.png` and quotes numbers from `results/*.csv`, but nothing generates it
automatically -- update it by hand when a result changes.

Every script can be run directly: `python scripts/<name>.py` (run from
inside `scripts/`, or the imports between files won't resolve). Run them in
the order shown above the first time; after that, only `data_processer.py`
needs re-running if the raw data changes -- everything downstream re-derives
its own numbers each time it runs (nothing is cached between scripts except
via the CSVs each one reads back from `results/`).

---

## `download_data.ps1`

Downloads the raw football-data.co.uk CSVs (22 divisions x 10 seasons = 220
files) into `data/raw/`, skipping any file that already exists. Run this
first, once. Writes `data/download_log.csv`. Paths are derived from the
script's own location, so the project folder can be moved.

## `data_processer.py`

Turns the raw CSVs into BOTH processed datasets every other script reads --
one run produces `odds_long.csv` (proportional de-vig) AND
`odds_long_shin.csv` (Shin de-vig), computed from the same raw odds per
bookmaker so only the normalization formula differs between them. Merges
the VC-Bet/BetVictor rebrand into one continuous expert identity. Replaces
the older two-script `process_data.py` + `process_data_shin.py` (now in
`scripts/deprecated/`) -- verified to produce byte-/value-identical output
before the switch. Deliberately minimal: no exploration/coverage report
(unlike the deprecated version), a hardcoded bookmaker-prefix set instead
of auto-detection from column names, and no defensive checks beyond the
numeric-parsing ones that are actually load-bearing (stray non-numeric odds
values in some leagues -- see `MERGE_ALIASES`/`AGGREGATE_PREFIXES` below).

| Function | What it does |
|---|---|
| `find_bookmakers_odds(columns)` | `{prefix}H/{prefix}D/{prefix}A` triples present in `columns`, checked against the hardcoded `KNOWN_BOOKMAKER_PREFIXES` set (not auto-detected) |
| `devig_normalize(h, d, a)` | Proportional normalization: p_i = (1/odds_i) / sum(1/odds_j) |
| `shin_normalize(h, d, a, n_iter=60)` | Shin (1992) de-vig, one match at a time via simple bisection (a plain Python loop, not vectorized -- intentionally simple over fast) |
| `load_one_file(path)` | One raw `<League>_<Season>.csv` -> two long-format odds frames (basic, Shin), from the same raw odds |
| `main()` | Runs `load_one_file` over every raw file, saves both CSVs -- nothing else |

Key constants: `AGGREGATE_PREFIXES` (Max/Avg columns to exclude, they're not
independent experts), `MERGE_ALIASES` (the VC/BV merge),
`KNOWN_BOOKMAKER_PREFIXES` (the hardcoded bookmaker list).

**Outputs:** `data/processed/odds_long.csv`, `data/processed/odds_long_shin.csv`

---

## `sleeping_experts.py` — *the foundation (algorithms/data-loading only)*

Generalizes Hedge/OGD/FTRL to the full bookmaker panel (26 experts, most
with PARTIAL match coverage) via the Sleeping Experts / Specialists
reduction. This is the shared library almost everything else imports from
(15 other active scripts) -- it does NOT run anything or produce any output
itself; the experiment that does is `sleeping_experts_experiment.py`.

| Function | What it does |
|---|---|
| `load_full_universe()` | Builds the (rounds x experts x outcomes) tensor + awake mask + season labels, sorted chronologically. Returns `(panel, P, awake, y, dates, seasons)` |
| `project_to_simplex(v)` | Euclidean projection onto the simplex |
| `per_round_expert_loss(P, y)` | Per-expert log-loss each round (NaN-safe via EPS clipping) |
| `run_hedge_sleeping(loss, awake, N)` | Sleeping-experts Hedge |
| `run_ogd_sleeping(loss, awake, N)` | Sleeping-experts OGD |
| `run_ftrl_sleeping(loss, awake, N)` | Sleeping-experts FTRL |
| `last_awake_weight(W, awake)` | Each expert's weight at the LAST round it was awake. The correct way to read a "final weight" out of any `run_*_sleeping` result -- `W[-1]` is 0 for every expert asleep in the final match (10 of 26 here, PSC included), see the function docstring |
| `evaluate(probs, y, mask=None)` | log-loss / Brier / RPS / accuracy, averaged over masked rounds |

**Outputs:** none (library file) -- has no `main()`, is never run directly.

**Exports used by other files:** `DATA_DIR`, `OUT_DIR`, `EPS`, `M`, `COLORS`,
`OUTCOME_INDEX`, `load_full_universe`, `per_round_expert_loss`,
`run_hedge_sleeping`, `run_ogd_sleeping`, `run_ftrl_sleeping`, `evaluate`,
`last_awake_weight` (imported by `sleeping_experts_experiment.py`,
`fixed_share.py`, `contextual_experts.py`).

---

## `sleeping_experts_experiment.py` — *the experiment*

Runs `sleeping_experts.py`'s algorithms over the full panel and builds the
main results table -- split out from `sleeping_experts.py` itself (in a
later session) so that file could stay a pure algorithms/data-loading
library with no side effects from importing it.

| Function | What it does |
|---|---|
| `cumulative_average(x)` | Running mean |
| `cumulative_average_masked(x, mask)` | Running mean over masked rounds, holding flat while asleep (for continuous plot lines) |
| `plot_comparison(...)` | Saves the two-panel comparison plot |
| `main()` | Runs everything, builds and saves `results_table.csv` (with the `single_season` warning flag), saves the plot |

**Outputs:** `results/results_table.csv`, `results/sleeping_experts_comparison.png`

---

## `calibration_analysis.py`

Diagnostic pass: are the predicted probabilities well-calibrated? Found the
favorite-longshot bias that motivated `calibration_correction.py`. Does NOT
fix anything itself.

| Function | What it does |
|---|---|
| `pooled_prob_outcome_pairs(probs, y, mask=None)` | Flattens (T,3) probabilities + outcomes into (predicted, hit) pairs, 3 per match |
| `calibration_table(p_flat, hit_flat, name)` | Bins predictions into `N_BINS` and computes mean predicted vs. actual frequency per bin |
| `expected_calibration_error(df)` | ECE: bin-count-weighted mean absolute gap |
| `_style_axes(ax)` | Shared matplotlib cosmetic styling (re-imported by `calibration_correction.py`) |
| `plot_reliability(tables, out_path)` | Saves the 2-panel reliability + gap-zoom plot |
| `main()` | Computes calibration tables/ECE for the 4 algorithms + `REFERENCE_BOOKMAKERS`, saves table + plot |

**Outputs:** `results/calibration_table.csv`, `results/calibration_reliability.png`

**Exports used by other files:** `REFERENCE_BOOKMAKERS`, `SERIES_COLORS`,
`pooled_prob_outcome_pairs`, `calibration_table`, `expected_calibration_error`, `_style_axes`.

---

## `calibration_correction.py`

The fix for the favorite-longshot bias: online temperature scaling as a
SECOND OCO stage on top of each mixture. Proper chronological train/val
split so the reported val numbers never saw the rate-selection step.

| Function | What it does |
|---|---|
| `temperature_calibrate(probs, y, lr_scale)` | Causal online temperature scaling via projected OGD on a convex 1-D loss |
| `select_lr_on_train(probs_train, y_train, lr_grid)` | Picks the best learning-rate scale using ONLY the train prefix |
| `run_series(name, probs, y, mask=None)` | Full fit + train/val evaluation pipeline for one series |
| `plot_val_comparison(val_data, out_path)` | Saves the raw-vs-calibrated val-split plot |
| `main()` | Runs `run_series` for the 4 algorithms + `REFERENCE_BOOKMAKERS`, saves table + plot |

**Outputs:** `results/calibration_correction_table.csv`, `results/calibration_correction.png`

**Exports used by other files:** `temperature_calibrate`, `select_lr_on_train`,
`TRAIN_FRACTION`, `LR_GRID` (imported by `final_ranking.py`).

---

## `final_ranking.py`

The single authoritative ranking table: raw AND calibrated metrics, both
computed the SAME way (full causal history), so they're directly comparable
— unlike `calibration_correction.py`'s val-only numbers.

| Function | What it does |
|---|---|
| `calibrate_full_history(probs, y, mask)` | Selects the calibration rate on a train-only prefix, then runs ONE causal pass over the FULL history (train+val) with that fixed rate |
| `main()` | Builds the Tier-A (coverage >= 70%) table, raw vs. calibrated, for 4 algorithms + qualifying bookmakers |
| `plot_slope(table, out_path)` | Saves the raw-vs-calibrated slope-chart plot |

**Outputs:** `results/final_ranking_table.csv`, `results/final_ranking.png`

**Exports used by other files:** `calibrate_full_history` (imported by
`significance_test.py`, `value_betting.py`, `value_betting_online_kelly.py`,
`market_devig_comparisons.py`, `contextual_experts.py`,
`pooling_window_comparison.py`, `calibration_per_season.py`), `COVERAGE_THRESHOLD`,
`RANK_COLORS` (reused by `market_devig_comparisons.py`).

---

## `significance_test.py`

Are the small log-loss differences between algorithms/bookmakers real, or
noise? Moving block bootstrap (accounts for serial correlation in
consecutive match losses, unlike a plain t-test).

| Function | What it does |
|---|---|
| `moving_block_bootstrap_test(diff, ...)` | The bootstrap engine: resamples contiguous blocks, returns mean diff / CI / p-value. **Reused by `value_betting.py`, `value_betting_online_kelly.py`, `market_devig_comparisons.py` and `contextual_experts.py`.** |
| `full_length_raw_logloss(probs, y, T, mask=None)` | Per-round log-loss, full T-length, NaN where masked out |
| `full_length_calibrated_logloss(probs, y, mask, T)` | Same, but calibrated (via `final_ranking.calibrate_full_history`) |
| `paired_diff(a, b)` | Aligns two per-round series on their common defined rounds, in chronological order |
| `main()` | Runs the 12-comparison battery (OGD vs. others, raw vs. calibrated, etc.) |

**Outputs:** `results/significance_test_table.csv`

**Key finding (current 10-year dataset):** every mixture algorithm
significantly beats the naive uniform average, and OGD significantly beats
both Hedge (p=0.026) and FTRL (p=0.012). PSC significantly beats OGD, raw
and calibrated. Calibration significantly helps all four algorithms.
(On the original, smaller dataset OGD was the only one to beat uniform --
that reversed when the data grew.)

---

## `value_betting.py` — *extension, consolidated*

Economic sanity check: would a better forecast actually make money, once
the bookmaker's margin (vig) is accounted for? Every value-betting
experiment in the project EXCEPT `value_betting_online_kelly.py` (which
swaps out the staking mechanism itself, not the training panel or
normalization, so it stays a separate file). Consolidates 6 previously
separate scripts (`value_betting.py`'s original 2-bookmaker test,
`value_betting_all_bookmakers.py`, `value_betting_closing_trained.py`,
`value_betting_training_panel_sweep.py`, `shin_vs_basic_value_betting.py`,
and `opening_closing_quality_transfer.py`'s value-betting half) into one
file, now in `scripts/deprecated/` -- see that folder's README for the exact
mapping. A leave-one-bookmaker-out design throughout (refits OGD excluding
the bookmaker being bet against) avoids the circularity of betting against a
bookmaker whose own price fed the mixture; an OPENING-side target further
restricts the training panel to other opening-only bookmakers, since none of
the closing-side information the full panel would otherwise include exists
yet at opening-bet time (see "Key finding" below for why this matters).

| Function | What it does |
|---|---|
| `load_match_order()` / `load_decimal_odds(book, match_ids)` | Chronological match ordering + raw decimal odds per bookmaker (`load_full_universe()` doesn't keep either) |
| `load_universe_from(csv_path)` / `split_by_market_phase(panel)` / `select_bookmakers(min_coverage)` | This file's own small copies of the universe-builder, opening/closing splitter, and coverage-based bookmaker filter (kept independent of `market_devig_comparisons.py`, which carries its own) |
| `kelly_fraction(p, odds)` / `simulate_kelly_betting(p_hat, odds, y, mask, kelly_mult)` | Full Kelly stake fraction; bets the highest-positive-EV outcome each round at `kelly_mult` x Kelly, returns bankroll growth factors |
| `fit_leave_out(panel, P, awake, y, indices, target_book, match_ids, exclude_target, apply_reduction)` | THE fitting routine behind every experiment below: OGD on `indices` (full panel / open_idx / close_idx), optionally excluding the target, causally calibrated. Replaces 5 near-identical copies of this pattern that used to be scattered across the merged files. `apply_reduction` preserves a real behavioral distinction the original scripts had: opening-only/closing-only fits filtered to rounds where >=1 of the small subset was awake, but the full-panel-minus-target ("mixed-trained") fits never did -- verified during the merge (see below) |
| `score_bet(growth, bet_placed, book_mask, fields)` | Shared bankroll + bootstrap-test scoring, one result row |
| `run_original_two` / `run_all_bookmakers_sweep` / `run_closing_trained_both_norms` / `run_bw_leakage_check` / `run_training_panel_sweep` / `run_shin_vs_basic` | The 6 experiments, each with its own plot function; see `main()` for the run order |

**Outputs:**
`results/value_betting_table.csv`, `.png` (experiment 1: original B365C/BW test) |
`results/value_betting_all_bookmakers_table.csv`, `.png` (experiment 2: 11-bookmaker sweep) |
`results/value_betting_closing_trained_table.csv`, `.png` (experiment 3: closing-on-closing, both norms) |
`results/opening_closing_quality_transfer.csv`, `.png` and `results/opening_value_betting_corrected.csv`, `.png` (experiment 2's correlation half lives in `market_devig_comparisons.py`; the BW leakage check itself, experiment 4, is here) |
`results/value_betting_training_panel_sweep.csv` (experiment 5: 108-row full grid) |
`results/shin_vs_basic_value_betting_table.csv`, `_significance.csv`, `.png` (experiment 6)

**Key finding:** no significant edge against B365C (a sharp bookmaker). The
original BW (a weaker, opening-side bookmaker) result — a large, significant
edge — is a **look-ahead-bias artifact**: excluding only BW's own column
still leaves every closing-side bookmaker (incl. BW's own closing
counterpart, BWC) in the refitting panel, information that doesn't actually
exist yet when only BW's opening price is posted. The honest, corrected
opening-only leave-one-out (`run_bw_leakage_check`) shows NO significant
edge either honest (12 bookmakers, BW excluded: raw -0.00010 p=0.169,
calibrated +0.00004 p=0.439) or circular (13 bookmakers, BW included: both
stages losses, not significant). Extended to all 11 sufficiently-covered
bookmakers (`run_all_bookmakers_sweep`), only 4 show a significant
calibrated edge: **positive** against WHC (+0.00049, p<0.001) and IW
(+0.00030, p=0.006); **negative** against PSC (-0.00024, p<0.001, sanity
check -- it's the #1 predictor overall, so betting against it with an
inferior model should lose) and PS (-0.00006, p=0.026) -- so the honest
story is "edge existence and sign depend on the specific bookmaker", not
"no edge vs. sharp, big edge vs. weak". The correlation check
(`market_devig_comparisons.run_quality_transfer_correlation`) finds no
reliable evidence that closing-side skill transfers to opening-side skill
either (13 pairs: Pearson r=-0.006). The full training-panel sweep
(`run_training_panel_sweep`) shows opening-trained betting against CLOSING
targets loses catastrophically (bankroll -> ~0, p<0.0001, all 5 closing
targets x both normalizations); the mixed-trained/opening-target leakage
artifact generalizes to every opening bookmaker, not just BW (apparent
"profits" up to hundreds of billions x); Shin vs. Basic never flips the
qualitative conclusion anywhere. `run_shin_vs_basic` (the 2 worst
algorithms, FTRL/Uniform average) finds the same pattern as
`market_devig_comparisons.py`'s core comparison: Shin significantly better
on raw log-growth in all 4 combinations, but only 1 of 4 remains significant
after calibration. Real-world account limits on winning bettors would
likely prevent sustained extraction of any edge found here either way — a
caveat, not part of the simulation itself.

**Verified during the merge:** every consolidated CSV matched its pre-merge
script's output value-for-value after sorting (tolerant of row order and
floating-point noise) EXCEPT for one real behavioral bug the merge caught:
an early draft applied the opening/closing-only "keep rounds where >=1 of
the subset is awake" reduction to the full-panel-minus-target case too,
which the original scripts never did -- `fit_leave_out`'s `apply_reduction`
parameter fixes this, and re-verification after the fix matched exactly.

---

## `fixed_share.py` — *extension*

Fixes a limitation found by inspecting `results_table.csv`: rarely-active
(often single-season) bookmakers can end up with a disproportionately large
final Hedge weight, because nothing erodes their weight while asleep.
Implements Fixed-Share (Herbster & Warmuth, 1998): a small per-round leak of
weight toward uniform, applied every time an expert is awake.

| Function | What it does |
|---|---|
| `run_hedge_fixed_share(loss, awake, N, alpha)` | Sleeping-experts Hedge + Fixed-Share leak. `alpha=0` is verified (via assertion in `main()`) to exactly reproduce `sleeping_experts.run_hedge_sleeping()` |
| `main()` | Runs a small alpha grid, reports overall log-loss and, per single-season bookmaker, the before/after weight it held the LAST TIME IT WAS AWAKE (via `sleeping_experts.last_awake_weight` -- reading `W[-1, k]` instead reported 0.0 for the 4 bookmakers whose only season ended before the end of the dataset, i.e. hid the worst cases) |

**Outputs:** `results/fixed_share_table.csv`, `results/fixed_share_weights.csv`

**Key finding:** alpha as small as 0.005-0.01 reins in the inflated weights
at a small but real cost to overall log-loss (0.99845 -> 0.99885, i.e.
slightly worse). Per-bookmaker: CLC 0.240 -> 0.064 and CL 0.222 -> 0.063 at
alpha=0.01; the four 2024/25-only bookmakers (1XB, 1XBC, BF, BFC) go from
0.164-0.177 down to exactly 0.100 = 1/10, i.e. parity within the 10-strong
awake set of their last round.

---

## `contextual_experts.py` — *extension*

Tests the original thesis idea directly: does learning SEPARATE weights per
league beat one pooled/global model? Kept in its own file deliberately (see
the module docstring) so the already-stable core files don't need to change
again.

| Function | What it does |
|---|---|
| `load_league_labels(match_ids)` | League code per match, aligned to `load_full_universe()`'s order |
| `load_match_order()` | Same helper as in `value_betting.py` (duplicated on purpose — no cross-dependency between the two extension files) |
| `run_league_specialist(P, awake, y, league_idx)` | Fits an independent OGD using only one league's rounds |
| `plot_league_comparison(table, out_path)` | Saves the global-vs-specialist dot plot |
| `main()` | For each league in `LEAGUE_NAMES` (the 6 top-flight leagues E0/SC0/I1/SP1/D1/F1 -- deliberately not all 22 divisions, see the comment above `LEAGUE_NAMES`; the module docstring's "10 leagues" wording is stale): fits a specialist, compares to the global (pooled) OGD on the same rounds, reports the top-3 bookmakers the specialist trusts most, then one overall bootstrap test pooling all leagues |

**Outputs:** `results/contextual_experts_table.csv`, `results/contextual_experts.png`

**Key finding:** specialization does NOT help — the pooled global model
significantly beats per-league specialists overall (mean diff +0.00049,
95% CI [+0.00005, +0.00091], p=0.031; only Scotland's specialist wins, by a
hair), plausibly because each league only has 2,231-3,800 matches (too little
for the online algorithm to converge well on its own) and the top-trusted
bookmakers barely differ by league anyway (PSC is #1 in 4 of the 6 leagues,
B365 in the other 2 -- i.e. the specialists rediscover roughly the same
experts the pooled model already trusts, rather than finding anything
league-specific). NOTE: the pre-fix version of this list read `W_l[-1]`,
which silently excluded PSC everywhere (it doesn't quote every league's final
match) and reported "B365/BW dominate" instead -- see
`sleeping_experts.last_awake_weight`.

---

## `calibration_per_season.py` — *extension*

Per-season (instead of global) temperature calibration: resets BOTH the
round counter (so eta_t stays fast/responsive) AND the temperature itself
(s back to 1.0) at the start of every season. Fixes the global fit's
inability to track season-to-season drift, at the cost of a repeated
cold-start and a smaller per-season sample.

| Function | What it does |
|---|---|
| `calibrate_per_season(probs, y, mask, seasons, season_order)` | Runs `calibration_correction.temperature_calibrate()` independently once per season, on that season's own slice only |
| `plot_temperatures(...)` | Saves the per-season fitted-T trend plot |
| `main()` | Builds raw vs. global vs. per-season table for the 4 algorithms + Tier-A bookmakers, bootstrap-tests per-season vs. raw and vs. global |

**Outputs:** `results/calibration_per_season_table.csv`,
`results/calibration_per_season_significance.csv`, `results/calibration_per_season.png`

**Exports used by other files:** `calibrate_per_season` (imported by `calibration_warmstart.py`).

**Key finding:** per-season calibration significantly beats raw, but is
statistically indistinguishable from the existing global calibration
overall.

---

## `calibration_warmstart.py` — *extension*

A third point between `calibration_correction.py`'s global fit and
`calibration_per_season.py`'s cold-start-per-season fit: resets only the
round counter at each season boundary (keeping eta_t responsive), but
carries the PREVIOUS season's fitted temperature forward as the next
season's starting point instead of resetting to T=1. Duplicates small
pieces of `calibration_correction.py` (`temperature_calibrate`,
`select_lr_on_train`) with an added starting-temperature parameter, rather
than modifying those already-relied-upon functions.

| Function | What it does |
|---|---|
| `temperature_calibrate_warmstart(probs, y, lr_scale, s_init)` | Same online temperature-scaling recipe as `calibration_correction.temperature_calibrate()`, but starts from `s_init` instead of a hardcoded 1.0 |
| `select_lr_on_train_warmstart(...)` | Learning-rate selection variant that evaluates candidates starting from `s_init` |
| `calibrate_per_season_warmstart(probs, y, mask, seasons, season_order)` | Per-season loop like `calibrate_per_season()`, but carries the ending `s` from each season into the next instead of resetting to 1.0 |
| `plot_temperature_comparison(...)` | Saves the cold-start-reset (dashed) vs. warm-start (solid) per-season temperature comparison plot |
| `main()` | Builds raw vs. global vs. per-season vs. warm-start table for the 4 algorithms + Tier-A bookmakers, bootstrap-tests warm-start vs. each of the other three |

**Outputs:** `results/calibration_warmstart_table.csv`,
`results/calibration_warmstart_significance.csv`, `results/calibration_warmstart.png`

**Key finding:** warm-start visibly removes the cold-start sawtooth in the
per-season T trajectory and beats raw significantly almost everywhere, but
lands statistically indistinguishable from BOTH the global and cold-start
per-season variants — a genuine compromise point, not a strict improvement
over either.

---

## `value_betting_online_kelly.py` — *extension*

Does it help to LEARN the Kelly stake fraction itself online (projected OGD
on the negative-log-wealth loss -- the "Online Portfolio Selection" framing,
Cover 1991) instead of plugging a point-estimate probability into the
closed-form Kelly formula each round? A third OCO stage on top of the
pipeline (mixture weights -> optional calibration -> now stake size), same
recipe as `run_ogd_sleeping`/`calibration_correction.temperature_calibrate`.
Runs the closing-on-closing comparison, both normalizations, online-learned
stake vs. classical Kelly on the identical bets (paired bootstrap test).

| Function | What it does |
|---|---|
| `simulate_online_kelly(p_hat, odds, y, mask, F_MAX=0.5)` | Same outcome-selection as `value_betting.simulate_kelly_betting`, but the stake f is learned online via projected OGD instead of the closed-form formula |
| `fit_closing_on_closing(...)` | Same OGD fit as `value_betting.run_closing_trained_both_norms`'s per-target fit (via `fit_leave_out` there), but returns the raw ingredients so both stake mechanisms can be applied to the same bets -- its own small copy, not imported, same reasoning as `value_betting.py`'s other small local copies |

**Outputs:** `results/value_betting_online_kelly_table.csv`,
`results/value_betting_online_kelly_significance.csv`,
`results/value_betting_online_kelly.png`

**Key finding:** the online-learned stake LOSES to the classical Kelly
formula almost everywhere (17 of 20 comparisons significantly worse, the
rest still worse just not significantly) -- it starts blind (f=0) and must
learn the stake purely from realized win/loss outcomes via a
conservative, worst-case-safe learning rate, while the classical formula
uses the already-available probability estimate directly with no
cold-start cost. The right place for OCO here is the forecasting stage,
not re-learning what the closed-form answer already gives for free.

---

## `value_betting_ogd_vs_ftrl.py` — *extension*

Head-to-head of the two mixture algorithms INSIDE the betting simulation.
Exists because `value_betting.py`'s experiments 1-5 use only OGD and its
experiment 6 only FTRL/Uniform, so the two overlap on exactly one valid cell
(B365C, Basic) — and even there they are two independent "is this edge
different from zero?" tests, never a paired OGD-vs-FTRL test. Same axis-
isolation reasoning as `value_betting_online_kelly.py` (vary one stage, hold
everything else fixed), hence its own file and its own small copy of the fit
routine.

| Function | What it does |
|---|---|
| `fit_leave_out_algo(..., algo, apply_reduction=True)` | `value_betting.fit_leave_out` with the algorithm as a parameter (that one hardcodes `run_ogd_sleeping`); everything else deliberately identical |
| `verify_against_canonical_sweep(table)` | Asserts the OGD rows reproduce `value_betting_all_bookmakers_table.csv` value-for-value (same fixed bootstrap seed) — what makes the FTRL half trustworthy: if OGD reproduces, the algorithm is the only difference |
| `plot_comparison(table, sig_table, out_path)` | Dot plot, both algorithms' calibrated edge per bookmaker |
| `main()` | The 11-bookmaker sweep (same per-target-phase design as `run_all_bookmakers_sweep`), both algorithms, raw + calibrated, plus the paired bootstrap |

The paired test is over **rounds, not bets**: the two algorithms disagree
about which rounds have positive EV, so per-bet pairing isn't available
(unlike `value_betting_online_kelly.py`, where both staking rules play the
identical bet). Pairs per-active-round log-growth, 0 on a no-bet round —
same device as `value_betting.run_shin_vs_basic`. **SIGN: the bootstrapped
quantity is log-GROWTH, so positive `mean_diff` = OGD better** (opposite of
the project's log-loss tables).

**Outputs:** `results/value_betting_ogd_vs_ftrl_table.csv`,
`_significance.csv`, `.png`

**Key finding:** OGD is ahead in 15 of 22 comparisons but the gap is
significant in only 4 — and all 4 are OGD wins, none FTRL (B365C raw
p=0.045, BWC raw p=0.020, WHC raw p=0.003, WHC calibrated p=0.015). The
effect is concentrated entirely on CLOSING-side targets (OGD ahead in 9 of
those 10 comparisons, all 4 significant ones among them); on opening-side
targets the two split 6-6 with nothing significant. So the forecasting-stage
ranking (OGD significantly beats FTRL on log-loss, p=0.012) does carry into
the economics, but weakly: the paired differences (~2-3e-05 log-growth per
round) are an order of magnitude smaller than the edges themselves, i.e.
which bookmaker you bet against matters far more than which of the two
algorithms you use. Note FTRL is absolutely better on some targets (IW
calibrated: +0.00034 vs OGD's +0.00030, both significant edges).

---

## `value_betting_by_band.py` — *extension*

Follow-up to `accuracy_by_band.py`: that file found the mixture's log-loss
advantage is concentrated in the high-confidence bands, so the obvious next
question is whether restricting the BETS to a good band makes money. Same
honest leave-one-out design as `value_betting.run_all_bookmakers_sweep`
(per-target-phase panel, quarter-Kelly, raw + calibrated), with the bets
bucketed by the probability assigned to the outcome actually backed -- not
the argmax, since value betting frequently backs something other than the
favourite.

| Function | What it does |
|---|---|
| `simulate_kelly_with_bands(...)` | `value_betting.simulate_kelly_betting` plus the backed outcome's own probability per bet (what assigns a bet to a band); selection/staking/growth identical line for line |
| `score_subset(growth, sel, fields)` | Bankroll + bootstrap over an arbitrary subset of placed bets |
| `main()` | Full-sample pass (every band, descriptive) AND a train/val pass: the band is chosen on the first 75% of each target's bets by train bankroll, then scored ONLY on the remaining bets |

The train/val half is the point of the file: with 10 bands x 11 bookmakers
x 2 stages, picking the best band post hoc guarantees a winner. Quote the
holdout number, not the full-sample one.

**Outputs:** `results/value_betting_by_band_table.csv`,
`_holdout.csv`, `results/value_betting_by_band.png`

**Key finding: band filtering does NOT help, and the way it fails is the
useful part.** Full sample: 10 of 99 calibrated band-cells are significant
-- about what 99 tests at 95% produce by chance (~5), with mixed signs (4
positive, 6 negative). Honest holdout: the train-chosen band beats the
unrestricted strategy on val in **6 of 11** targets, i.e. a coin flip; val
bankroll > 1 in 5 of 11 vs 6 of 11 unrestricted; the only significant val
result is BWC at **p=0.025 in the LOSING direction**. Decisive detail: the
train-chosen bands are scattered across the whole spectrum (0-10%, 10-20%,
30-40% x3, 40-50%, 50-60%, 60-70%, 70-80% x2, 80-90%) -- if a genuinely
better probability region existed, the same band would keep being selected.
What band filtering actually does is cut bet counts ~6x and compress
outcomes toward 1 (PS: 0.39 -> 0.96, WH: 1.70 -> 1.11), i.e. variance
reduction, not edge discovery. B365C is the snooping failure mode in
miniature: train picked the 0-10% band, which contained 6 val bets.

---

## `outcome_class_accuracy.py` — *extension*

Per-outcome-class (1 / X / 2) breakdown. Every other metric averages over
the three outcome classes; this asks which algorithm/bookmaker is actually
best at calling each specific outcome as its single most likely one
(per-class recall). Restricted to Tier-A bookmakers (coverage >= 70%).

| Function | What it does |
|---|---|
| `per_class_metrics(probs, y, mask=None)` | One row per outcome class, over rounds whose TRUE outcome was that class: accuracy, log-loss, Brier |
| `plot_by_class(table, out_path)` | Saves the 3-panel accuracy-by-class plot |
| `main()` | Runs the 4 algorithms + Tier-A bookmakers, saves table + plot |

**Outputs:** `results/outcome_class_accuracy_table.csv`, `results/outcome_class_accuracy.png`

**Key finding:** draws are essentially unpredictable as an argmax pick --
the best series (PSC) picks 0.695% of actual draws correctly, OGD 0.299%,
despite draws being ~26% of matches. Best at home wins: BW (82.8%); at
away wins: PSC (49.3%).

---

## `accuracy_by_band.py` — *extension*

Every pooled comparison in the project averages over the whole probability
spectrum. This file asks whether a null result there hides a LOCALISED one:
is there a probability range where the mixture is clearly better (or worse)
than the bookmakers? Three views, Tier A only (4 algorithms + coverage>=70%
bookmakers).

| Function | What it does |
|---|---|
| `reliability_by_band(name, probs, y, mask)` | View A: pooled (predicted, hit) pairs bucketed into 10 fixed probability bands; per band n / mean predicted / realised frequency / gap. Same construction as `calibration_analysis.py`, but per band and for every Tier-A series rather than the 2 `REFERENCE_BOOKMAKERS` |
| `accuracy_by_confidence(name, probs, y, mask)` | View B: the accuracy question proper -- when the top pick carries 40-50% / 50-60% / ... confidence, how often is it right? (3 outcomes, so the top pick is always >= 1/3 and the bands start at 0.33) |
| `main()` | Builds A and B, then view C: OGD vs each Tier-A bookmaker, per-round log-loss, restricted to rounds where OGD's top pick falls in that confidence band -- the paired test that decides whether A/B's apparent differences are real |

`MIN_BAND_N = 200` flags thin bands rather than dropping them.

**Outputs:** `results/accuracy_by_band_reliability.csv`,
`_confidence.csv`, `_significance.csv`, `results/accuracy_by_band.png`

**Key finding:** the three views disagree, and that disagreement IS the
finding. (A) On calibration a bookmaker wins in **all 10 bands** -- the
mixtures are systematically WORSE calibrated than individual bookmakers
everywhere, because linear pooling shrinks probabilities toward the middle
(which is exactly what the temperature-scaling stage later undoes). (B) On
accuracy the series are indistinguishable, and where they aren't the signal
is misleading: IW has the best accuracy in almost every band yet the WORST
Tier-A log-loss -- and the per-band accuracies aren't on common rounds, so
coverage differences confound them. (C) On the properly paired log-loss
test, OGD beats every bookmaker except PSC in essentially every band:
**31 of 42 band x bookmaker cells significant for OGD, 0 against it except
PSC** (significant in all 6 bands), 5 ties, all in the two thinnest
high-confidence bands. Both margins GROW with confidence (OGD vs IW:
-0.0027 at 33-40% to -0.0079 at 80-100%; PSC vs OGD: +0.0012 to +0.0065),
so the confident predictions are where everything is decided, while the
33-50% bands -- around 60% of all matches -- are where every method is
nearly identical.

---

## `outcome_class_comparison.py` — *extension*

`outcome_class_accuracy.py` breaks accuracy/log-loss/Brier down by outcome
class but never tests whether the gaps it prints are real. This file adds
the paired bootstrap, the same way `accuracy_by_band.py` did for probability
bands: stratify by the TRUE outcome, then compare series within each stratum
on rounds where both are defined.

| Function | What it does |
|---|---|
| `describe(name, probs, y, mask, kind)` | Per class: log-loss, Brier, recall (of the actual home wins, how many it called) AND precision (of the times it called home, how often it was right -- not computed anywhere else in the project), plus the mean probability the series assigns to that class overall |
| `compare(a, ll_a, b, ll_b, y, cls, label)` | Paired bootstrap of per-round log-loss restricted to one class, or pooled when `cls=None` |
| `main()` | View A descriptive; view B algorithm vs algorithm; view C OGD vs every Tier-A bookmaker; then the tilt check |

**THE TRAP THIS FILE EXISTS TO HANDLE:** restricted to rounds that ended in
a home win, the log-loss is `-log(p_H)`, so a series that merely leans
toward home WINS that stratum and loses the other two by about as much.
Per-class log-loss measures tilt as much as skill. Every per-class
comparison is therefore paired with the same comparison pooled, and flagged
`consistent_with_pooled` -- a per-class win that points the opposite way to
the pooled result is a lean, not an advantage.

**Outputs:** `results/outcome_class_comparison_descriptive.csv`,
`_significance.csv`, `results/outcome_class_comparison.png`

**Key findings:**
1. **OGD's edge over Hedge/FTRL is entirely a home/away split, not uniform
   skill**: OGD significantly better on home wins (-0.00112 / -0.00118,
   p<0.001), significantly WORSE on away wins (+0.00119 / +0.00125,
   p<0.001), tied on draws. Mechanism measured directly: OGD assigns on
   average +0.0003 more probability to home and -0.0004 less to away than
   Hedge/FTRL. Its pooled win comes from home wins being the commonest
   outcome (33,280 vs 23,224 rounds). The tilt is toward the TRUE base rates
   though (home 43.46% actual vs ~43.0% predicted by all four; OGD least
   biased), so it is a correction, not a defect.
2. **PSC's advantage over OGD is entirely in home wins** (+0.00521 there,
   tie on draws) -- and **OGD significantly BEATS PSC on away wins**
   (-0.00271). The one bookmaker that beats the mixture overall does not
   beat it everywhere.
3. **Away wins are where the mixture is most reliably better than
   bookmakers** (OGD ahead of 5 of 7, zero losses); **draws are where the
   margins are biggest but mixed** (up to -0.0087 vs B365, but IW and WH
   beat OGD there); **home wins are the weakest** (3 wins, 3 losses).
4. **"Draws are unpredictable" needs qualifying.** Recall is indeed
   negligible (0.1-0.7%), but PRECISION is 35-56% against a 26.2% base rate
   for every single series -- even the weakest estimate (PSC, 35.5% on
   n=369) sits 3.7 standard errors above the base rate. A draw is almost
   never anyone's single most likely outcome; that is not the same as being
   unpredictable when it is called.
5. 25 of 33 per-class comparisons are significant, but 7 of them point the
   opposite way to their pooled comparison -- i.e. tilt, not skill.

---

## `value_betting_sharp_anchor.py` — *extension*

The natural challenge to the whole thesis, and the standard professional
recipe: PSC is the best single forecaster in the panel, so skip the learning
and bet PSC's de-vigged price against the softer bookmakers. If that earns
more than the OCO mixture, the mixture is economically redundant however
good its log-loss.

**Timing constraint -- an anchor may only bet its OWN phase.** PSC is a
closing price, so using it against an OPENING price is the same look-ahead
bias `run_bw_leakage_check` found and fixed. So the file runs two separate
anchors against same-phase targets only (coverage floor 15%, single-season
excluded): **PSC (closing)** vs B365C, BWC, VC_BVC, WHC, IWC, BFEC, and
**VC_BV (opening)** vs B365, BW, PS, WH, IW, LB, BFE. VC_BV is the best
Tier-A opening bookmaker by log-loss; LB scores better still but covers 27%
of matches over 3 seasons, too thin to anchor on. The mixture is likewise
restricted to the anchor's phase on the opening side.

Four forecasters on the identical opportunity set (rounds where the anchor,
the target and its odds all exist), quarter-Kelly throughout: anchor raw,
anchor calibrated, OGD raw, OGD calibrated (target excluded). Anchor vs OGD
is paired per ACTIVE ROUND on log-growth, since the four place different
bets.

A weaker assumption on the opening side, worth stating in the write-up: all
closing prices settle near kickoff, so treating them as simultaneous is
safe, whereas different bookmakers OPEN at different times -- the opening
anchor therefore assumes opening prices are mutually available, which is the
same assumption `value_betting.py`'s opening-only panels already make.

**Outputs:** `results/value_betting_sharp_anchor_table.csv`,
`_significance.csv`, `results/value_betting_sharp_anchor.png`

**Key finding: the anchor strategy works on the CLOSING side and not on the
OPENING side, and the size of the anchor's dominance explains why.**

- **Closing anchor (PSC)**: ahead of the calibrated mixture on **6 of 6
  targets**, sometimes by a lot (WHC 88.7x vs 34.2x, IWC 8.2x vs 3.8x, BWC
  2.7x vs 1.5x) -- yet **no paired difference is significant** (best
  p=0.052). Sign test across targets p=0.031, but the six share matches and
  the same PSC forecasts, so treat as suggestive. Standalone, PSC turns a
  significant profit against IWC (+0.00034, p=0.002) and WHC (+0.00057,
  p<0.001).
- **Opening anchor (VC_BV)**: ahead on only **4 of 7** targets (sign test
  p=1.0), nothing significant either way -- a coin flip. And the opening side
  is barely profitable for anyone: of 7 targets only IW pays (anchor 14.0x,
  mixture 8.2x, both p<0.01), PS loses significantly for both, and the rest
  hover around 1.0.

The mechanism is the anchor's margin over the mixture in its own phase:
PSC beats OGD (closing) by 0.00031 in log-loss, VC_BV beats OGD (opening) by
only 0.00008 -- four times smaller. Anchoring pays to the extent one expert
genuinely dominates its panel, which is the same lesson as the
uniform-baseline and per-panel ranking results. Note also that the mixture
ALREADY contains the anchor (PSC ~0.14 weight on a closing target, see
`value_betting_circular.py`), so this is "anchor alone vs a blend that
dilutes it", not "single vs learned" in a clean sense.

Caveat for the write-up: WHC, IWC and IW are the softest books in the panel
and precisely the accounts that get limited first, which the simulation does
not model.

---

## `value_betting_circular.py` — *extension*

What happens if the leave-one-out is simply skipped? `run_bw_leakage_check`
tests that for one bookmaker; this does it for all 11 as a controlled A/B --
same panel, same Kelly, same calibration, only `exclude_target` flipped. The
honest side is read from `value_betting_all_bookmakers_table.csv` rather
than refitted, so the comparison is against the published numbers.

| Function | What it does |
|---|---|
| `fit_including_target(...)` | `value_betting.fit_leave_out` plus the fitted weight matrix, so the size of the effect can be tied to the mean weight the mixture puts on the target itself |
| `plot_circular(merged, out_path)` | Slope plot, honest vs circular edge per bookmaker, labelled with that weight |

**Outputs:** `results/value_betting_circular_table.csv`,
`_vs_honest.csv`, `results/value_betting_circular.png`

**Key finding -- circularity DESTROYS the experiment rather than inflating
it, and that is the opposite of the project's other leakage bug.** The
look-ahead bug inflated returns (future information). Circularity cannot:
the target's own de-vigged price enters the mixture, so `p_hat` is pulled
toward the price being bet against, while the bet still needs
`p_hat * odds > 1` against that same bookmaker's VIGGED odds. In the limit
the two cancel and no bet ever clears.

Measured: **bets fell in 11 of 11 targets, median -35%**, and the size of
the fall tracks the weight the mixture puts on the target almost perfectly
(**Pearson r = -0.83**). That weight is the striking number here -- on an
opening-only panel OGD concentrates hard, giving the target a mean weight of
**0.72-0.89** (B365 0.887, BW 0.823, VC_BV 0.806). So "betting against B365"
becomes "betting against B365 with a forecast that is 89% B365": bets drop
68% and the bankroll goes 0.36 -> 0.95, i.e. flat. Closing targets sit on
the full 25-expert panel, get far less weight (WHC 0.035, BWC 0.046), and
are barely affected (bets -3%, edge unchanged). Two edges flip sign outright
(BW, WH -- both around 0.8 weight), and the significant-result count drops
from 4 to 3. Conclusion for the write-up: the leave-one-out is not there to
stop the numbers looking too good, it is there because without it you are
testing a price against a blend of itself.

---

## `outcome_class_accuracy_test.py` — *extension*

The last piece of the outcome-class work: `outcome_class_accuracy.py` and
`outcome_class_comparison.py` both PRINT per-class accuracy numbers, neither
TESTS them. Pooled accuracy was already shown to carry no signal at all
(0 of 18 significant, `opening_vs_closing_vs_uniform*.csv`); this file
checks whether splitting into 1 / X / 2 rescues it.

| Function | What it does |
|---|---|
| `recall_series(probs, y, mask, cls)` | Full-length 0/1 "picked this class" indicator, NaN outside the stratum -- shaped for `paired_diff` |
| `precision_ci(probs, y, mask, cls, rng)` | Per-series bootstrap CI for precision + whether it clears the class base rate |
| `main()` | Recall: paired tests, OGD vs the other 3 algorithms and vs every Tier-A bookmaker, per class. Precision: CI per series x class |

**Why the two halves are treated differently:** RECALL is pairable (all
series are defined on the same stratum). PRECISION is not -- two series call
draws on DIFFERENT matches, so there is no common denominator, the same
obstacle as `value_betting_ogd_vs_ftrl.py` hits with differing bets. So
precision gets a per-series CI and is compared against the class BASE RATE
instead, which is the sharpest form of "does this call mean anything".

**SIGN:** higher is better here, unlike every log-loss table in the project;
`verdict` names the winner outright.

**Outputs:** `results/outcome_class_accuracy_recall_tests.csv`,
`_precision_ci.csv`, `results/outcome_class_accuracy_test.png`

**Key finding -- per-class accuracy IS meaningful, but not as a way to
rank methods.**
- **Every series' calls beat chance, decisively: 32 of 33 series x class
  cells clear their base rate at 95%.** Home 51% vs a 43.5% base rate, Away
  **48.6% vs 30.3%** (a 1.6x lift, the most informative call of the three),
  Draw 35-56% vs 26.2%. The single failure is Uniform average on draws
  (36.0%, CI low 25.8% against a 26.2% base rate) -- it calls draws least
  often of the four algorithms (89 times).
- **But it still cannot separate the methods.** 23 of 33 recall comparisons
  come back significant, which looks like a lot after the pooled 0 of 18 --
  except recall differences are driven by HOW OFTEN each series picks that
  class, not by skill (OGD calls draws 161 times to B365's 59, so it
  "wins" draw recall and "loses" home recall; the same tilt artifact
  `outcome_class_comparison.py` documents for per-class log-loss). On
  precision, which is the fair comparison, the CIs overlap almost completely
  for Home and Away, and are far too wide for Draw (n = 45-369).

---

## `information_arrival.py` — *extension*

`market_devig_comparisons.py` established that a closing-only mixture beats
an opening-only one by ~0.0034 log-loss and read that as "the market absorbs
information before kickoff". That is an AVERAGE. This file asks whether it
is uniform or concentrated in the matches where the price actually moved --
i.e. whether the average hides a mechanism. Two stratifiers, both already
derivable from data the project has:

| Stratifier | How |
|---|---|
| `movement_per_match` | Total-variation distance between each bookmaker's opening and closing probability vector, averaged over the 13 available pairs (needs >= `MIN_PAIRS`) |
| `z_anomaly_per_match` | Shin's fitted insider-trading proportion (`shin_z`, computed by `data_processer.py` and never before analysed as a signal), as an anomaly WITHIN bookmaker -- raw z tracks each bookmaker's overround, so the raw value would mostly encode who quoted |

Only OGD and the uniform average are fitted (not Hedge/FTRL): the question
is about the market, not about which algorithm wins.

**Outputs:** `results/information_arrival_strata.csv`,
`_significance.csv`, `results/information_arrival.png`

**Key finding: the closing-odds advantage is almost entirely a top-quintile
phenomenon.** Across movement quintiles the closing-minus-opening gap goes
-0.00044 (Q1, **p=0.052, not significant**), -0.00072, -0.00200, -0.00452,
**-0.01158** (Q5) -- a **26x** gradient. In the 20% of matches where the
line barely moved there is no measurable closing advantage at all. Shin's z
is a much weaker stratifier by comparison (-0.00293 to -0.00410, 1.4x,
all significant).

**Important qualification, state it in the write-up:** movement is measured
from the same closing prices the closing panel is built on, so Q1's null is
partly MECHANICAL -- if the two prices barely differ, the two mixtures
barely can. The non-trivial part is the direction at the other end: when
prices move a lot they could move either way, and the later price is
systematically the better one by 0.0116. That is the efficient-markets claim
localised to where it actually operates.

Useful control: raw difficulty is NOT monotonic across the strata
(full-panel log-loss runs 0.99120, 1.00510, 0.99877, 1.00241, 0.99683) while
the closing advantage is, so the gradient is not a "harder matches, bigger
differences" artifact.

What odds alone cannot do: separate insider from public information (a big
move may be a leaked lineup or an announced injury). The claim is about
information ARRIVING, not its nature -- framed as market microstructure
(Kyle 1985; Glosten & Milgrom 1985; Shin 1992), not "insider detection".

---

## `no_pinnacle_panel.py` — *extension*

Answers the obvious objection to "the mixture beats every bookmaker except
PSC": that the mixture is mostly reproducing Pinnacle, which is one of the
experts inside it. Drops BOTH Pinnacle columns (PS and PSC) from the panel,
refits all four algorithms from scratch on what is left, and ranks them
against the surviving bookmakers on log-loss.

Runs over `PANELS`, currently two: the full panel (24 experts survive,
coverage floor 70%) and the closing-only panel (12 survive, floor 40%). The
second exists because `best_configuration.py` found closing-only to be the
project's best configuration, and its floor has to be lower -- once PSC is
gone NOTHING in that panel reaches 70% (next best B365C at 69.6%).

Uses the same honest-comparison discipline as `best_configuration.py`, which
matters here: **every series is scored on ONE common round set** (the
intersection where all of them are defined) plus a chronological
`TRAIN_FRACTION=0.75` split, and the paired bootstrap is restricted to that
set. An earlier version scored each series on its own coverage, and that
loose version produced a win over B365C on the closing panel that did NOT
survive -- it was an artifact of asymmetric coverage.

**Standalone on purpose** -- writes only its own outputs, changes nothing in
the main pipeline (which keeps all 26 experts), depends on no other script's
results.

**Outputs:** `results/no_pinnacle_ranking.csv`, `_significance.csv`,
`results/no_pinnacle.png`, `results/no_pinnacle_closing.png` (both panels'
rows are concatenated into the two CSVs, keyed by a `panel` column)

**Key finding 1, full panel: without Pinnacle the mixture is unambiguously
the best forecaster available.** On 54,877 common rounds all four algorithms
rank above all five surviving Tier-A bookmakers -- Hedge 0.99772, FTRL
0.99778, OGD 0.99782, uniform 0.99837, calibrated, against VC_BV's 0.99987 --
and **OGD significantly beats 5 of 5** (margins -0.00181 to -0.00315, all
p<0.001), much wider than anything it manages against PSC on the full panel.
OGD is also first on the held-out tail. Every round still has at least one
quote after the drop (100% coverage). Reading: the "a bookmaker beats us"
caveat is about one exceptional odds-setter, not about the method, and for
anyone who cannot access Pinnacle -- most bettors in most jurisdictions --
the OCO mixture is the best forecast obtainable from the rest of the market.

**Key finding 2, closing-only panel: remove Pinnacle from the BEST
configuration and the learning disappears.** 12 experts, only 32,210 common
rounds. OGD ranks #1 (0.99919 calibrated) but significantly beats only **3 of
5** (WHC -0.00088, IWC -0.00077, BWC -0.00051); B365C (p=0.505) and VC_BVC
(p=0.284) are ties, and VC_BVC is ahead of OGD on the held-out tail. The
telling number is that OGD (0.99919) is essentially IDENTICAL to the plain
uniform average (0.99921), with Hedge (0.99923) and FTRL (0.99932) behind it.
This does not contradict finding 1, it localises it: without Pinnacle the
closing bookmakers are too similar to each other for weighting to have any
quality difference to learn, which is the same mechanism seen in every
homogeneous panel in this project (see `market_devig_comparisons.py`). Worth
saying plainly in the write-up: the best configuration is best partly BECAUSE
it contains the market's best odds-setter.

Do NOT compare a with-Pinnacle number against a without-Pinnacle one
directly -- the round sets differ (76,584 vs 54,877 vs 32,210), so a
"Pinnacle's contribution is X" sentence built that way is not measuring what
it claims. An earlier draft of this file's write-up did exactly that.

---

## `deployment_test.py` — *extension*

The question every other ranking in the project sidesteps: **take what the
model knew at the end of 2024, run it through 2025, does it beat the market
that exists then?** Everything else scores the mixture over the same history it
learned on — legitimate for OCO, since the algorithm is causal and cumulative
loss is what a regret bound is about, but it is not what a supervisor or a
bettor is asking.

Real calendar cut at `CUT = 2025-01-01`: 65,009 train rounds, 11,574 test,
8,151 common. Two readings of "trained on 2016–2024", both reported:

| Variant | What it does |
|---|---|
| `online` | keeps updating through the test window (still causal; what you would deploy) |
| `frozen` | weights locked at the cut; still projected onto each round's awake subset, which is required to predict at all, not a learning step |

The calibrator matches the variant: its rate is always chosen on TRAIN rounds
only, then either kept live or frozen at the cut temperature.

**Key finding: beats every opening price, ties every closing price.** Opening
margins −0.0022 to −0.0065, all p≤0.007. Closing: B365C p=0.184, BWC p=0.334,
BFEC p=0.051 — ties in the best variant. `online` beats `frozen` consistently
(1.00109 vs 1.00151), so the honest phrasing is "runs continuously", not "was
trained until 2024".

**Two things that must be said whenever this is quoted.** Pinnacle is NOT in
the main table: `ASIDE = ("PS", "PSC")` holds it out because its coverage falls
to 59% after 2025 and it stops quoting on 2026-01-14. Forcing it into the
shared round set cuts 8,151 → 2,860 and leaves nothing significant anywhere,
so it is tested one-to-one on its own overlap instead — **we beat PS
significantly** (−0.0022 to −0.0023, p 0.006–0.015, 6,857 rounds) and **tie
PSC** (p 0.149–0.549, 6,892). And the best forecaster in that window is not a
bookmaker at all but **Betfair Exchange closing** (BFEC, 96% coverage): a
market price with no house margin.

**Outputs:** `results/deployment_test.csv`, `_significance.csv`,
`results/deployment_test.png`

---

## `recent_window.py` — *extension*

Every forecasting experiment in the project, rerun on the last five seasons
(2021/22–2025/26, 38,749 rounds), from scratch rather than sliced out of a
10-year run. The reason is not "newer data" but **which bookmakers survive a
coverage floor**: over 10 seasons exactly one closing column clears 70% (PSC),
because every other closing column starts around 2019/20 and B365C misses at
69.6%. So the 10-season headline "beats every bookmaker but PSC" is mostly a
statement about OPENING prices. In the 5-season window B365C, BWC, VC_BVC and
WHC all clear it, and the comparison people actually care about becomes
possible.

Sections A–H mirror `final_ranking`, `significance_test`,
`market_devig_comparisons`, `best_configuration` and `no_pinnacle_panel`, plus
one that exists only here:

- **Section H, `section_live_market`** — the mixture against bookmakers alive
  in the chronological tail only, with the floor recomputed inside that window.
  Algorithms are still fitted on the FULL history; only scoring is restricted.

**Key findings.** OGD is #2 of 14 (0.99488 vs PSC 0.99439); the closing-only
panel stays significantly better than the full one; the PSC gap shrinks to
+0.00029 (p=0.032) from +0.00044. **Two findings REVERSE** and must be reported:
Hedge and FTRL lose to the uniform average in *all* panels here, not only the
restricted ones, and log-loss and Brier stop agreeing on the ordering.

**Outputs:** `results/recent_window_ranking.csv`, `_significance.csv`,
`_panels.csv`, `_configs.csv`, `_no_pinnacle.csv`, `_live_market.csv`,
`results/recent_window.png`

---

## `recent_window_value_betting.py` — *extension*

The betting side of the same window, phase-matched (closing panel → closing
target, opening panel → opening target — the second is a causality
requirement). The mixed panel is also run against closing targets as the
comparison point showing the phase match is not cosmetic.

**SIGN WARNING:** this file bootstraps log-GROWTH per bet, where HIGHER is
better — the opposite of every log-loss table in the project.

**Outputs:** `results/recent_window_value_betting.csv`

---

## `best_configuration.py` — *extension*

Four wins had been measured separately and never combined: the corrected OGD
step (`learning_rate_study.py`), the closing-only panel beating the full one
(`market_devig_comparisons.py`), Shin de-vig beating proportional on raw
log-loss (same), and calibration. The shipped pipeline runs the FULL panel
with BASIC normalization, i.e. it leaves two of the four on the table. This
file stacks them and asks the one question that can change the project's
headline: **does any configuration reach PSC on a like-for-like comparison?**

| Config | Panel | De-vig |
|---|---|---|
| `baseline` | full 26 | basic |
| `closing+basic` | closing-only 13 | basic |
| `full+Shin` | full 26 | Shin |
| `A` | closing-only 13 | Shin |
| `B` | A minus single-season bookmakers | Shin |

All use OGD at the tuned step, each reported raw and causally calibrated.

Two things that make this honest, both of which the project has been bitten
by before:

1. **Every configuration is scored on the SAME rounds** -- the intersection
   of every config's coverage with PSC's, 71,818 rounds. The panels have
   different coverage and a log-loss over a different match set is not a
   comparable number. Fitting still uses each config's full support; only
   scoring is restricted.
2. **Choosing a configuration by looking at these numbers IS selection**, so
   everything is reported twice: full sample (descriptive) and a held-out
   chronological 25% tail the choice never saw.

| Function | What it does |
|---|---|
| `single_season(panel, awake, seasons)` | Bookmakers active in exactly one season -- confounded with that season's difficulty, and the ones holding inflated frozen weights |
| `fit_config(P, awake, y, indices)` | OGD at the tuned step on a column subset, then `calibrate_full_history`; returns per-round raw and calibrated log-loss at FULL length, NaN where the subset has nobody awake |
| `plot_configs(table, out_path)` | Held-out scores as a dot plot with PSC as a dashed reference line |

**Outputs:** `results/best_configuration.csv`,
`results/best_configuration_significance.csv`,
`results/best_configuration.png`

**Key finding: the closing-only panel makes the mixture statistically TIED
with PSC, and the panel change does all the work.** Calibrated, same 71,818
rounds: PSC 0.99685, A 0.99692, closing+basic 0.99693, baseline 0.99728,
full+Shin 0.99731. Paired tests:

- A vs baseline **-0.00036, p<0.001**; closing+basic vs baseline -0.00034,
  p<0.001. The panel is the whole story.
- full+Shin vs baseline +0.00004, **p=0.181** -- Shin alone adds NOTHING
  measurable once calibration is in place, exactly as `market_devig_
  comparisons.py` predicts (both correct the same favorite-longshot bias).
- vs PSC: baseline loses significantly (+0.00043, p<0.001), **A ties
  (+0.00007, p=0.234)**, closing+basic ties (+0.00009, p=0.143).
- B (A minus single-season) is marginally worse than A on the held-out tail
  and was rejected.

**State the limits when quoting this.** The point estimate still favours PSC;
"tied" means the difference stopped being significant, not that the mixture
won. On the held-out tail all six rows sit within 0.0001 of each other, so
the tail confirms the tie rather than picking a winner. And every closing
configuration has PSC INSIDE the mixture, so this reads "the blend reaches
its own best member", not "we reached it from outside" --
`no_pinnacle_panel.py` is the experiment that removes it, and on this panel
it finds the learning gain gone.

---

## `learning_rate_study.py` — *extension*

Are the shipped learning rates any good? Two things worth questioning: (a)
`M = log(1/EPS) ~ 13.82` bounds a single round's per-expert loss assuming a
bookmaker might price the realised outcome at 1e-6, whereas real de-vigged
football probabilities bottom out near 1-2%, so every `eta ~ 1/M` in the
project is several times smaller than an honest bound would allow; (b) the
project linearises the loss, which only admits generic O(sqrt(T)) rates,
while `-log(<w, p>)` is EXP-CONCAVE and the Aggregating Algorithm / Bayesian
mixture (Vovk 1990) attains regret <= ln N at a CONSTANT eta = 1.

| Function | What it does |
|---|---|
| `report_loss_range(loss, awake)` | Measures how conservative M actually is against the observed loss distribution |
| `hedge_scaled(loss, awake, N, mult, const_eta)` | Hedge with the rate multiplied, or replaced by a constant (`const_eta=1` is the Bayesian update `w_i <- w_i * p_i(y_t)`). Carried in LOG weight space -- mandatory, see below |
| `ogd_scaled(loss, awake, N, mult)` | OGD with the step size multiplied |

**Both sweeps assert they reproduce the shipped algorithms at mult=1.0**, and
that check earned its keep: an earlier draft of `hedge_scaled` rescaled the
weight vector after each update to avoid underflow at eta=1, which is
precisely the bug `run_hedge_sleeping`'s docstring warns about (the
persistent state must never be renormalised, only the awake-restricted
prediction copy). It silently produced a DIFFERENT algorithm and a
completely different — and flattering — set of conclusions. The fix is log
weight space with a log-sum-exp shift applied to the prediction only.

**Outputs:** `results/learning_rate_study.csv`,
`results/learning_rate_study.png`

**Key findings:**
1. **M is conservative by roughly 3-4x**, as suspected -- but that does not
   translate into free performance for every algorithm.
2. **CORRECTED finding: Hedge and FTRL were NOT already optimal at x1 -- that
   was the grid's lower boundary, not an interior minimum.** The original
   grid (`RATE_MULTIPLIERS`, 1..1000) never looked below x1, so "x1 wins on
   held-out and every increase is worse" was true but incomplete: nobody had
   checked whether DECREASING the multiplier helped. It does. Extending the
   grid downward (`HEDGE_FTRL_MULTIPLIERS`, now 0.001..1000) found genuine
   interior minima at **x0.25 for Hedge** (train 0.998118, held-out
   0.998831, vs 0.998128/0.998962 at x1 -- +0.00013 held-out) and **x0.05 for
   FTRL** (train 0.998117, held-out 0.998889, vs 0.998160/0.999072 at x1 --
   +0.00018 held-out), now shipped as `HEDGE_STEP_MULTIPLIER` /
   `FTRL_STEP_MULTIPLIER` in `sleeping_experts.py`. **Lesson for any future
   sweep in this project: an answer sitting on a grid endpoint is not
   evidence of an optimum, only evidence that the grid needs extending.**
3. **OGD's step size is far too small.** Held-out log-loss improves
   monotonically from 0.99803 at x1 to **0.99765 at x100**, then degrades
   (x200 0.99778, x500 0.99858, x1000 0.99895). x100 is a genuine interior
   optimum (verified on a finer grid too: flat plateau x75-x110, all giving
   held-out 0.99765) and train and held-out agree on it, so the choice is
   not snooped. The gain, **+0.00038 held-out, is comparable to the entire
   OGD-vs-uniform advantage (0.00061) and to the benefit of the whole
   calibration stage (~0.00065)** -- available for the cost of one constant.
4. **The theoretically superior Bayesian/constant-eta variant performs
   BADLY here**: held-out 1.0007-1.0011 for eta in [0.1, 2], i.e. worse than
   uniform averaging (0.99882). The regret bound is about a STATIC
   comparator, and this panel is not static -- PSC's coverage collapses from
   ~100% to 38.8% in the final season, which lands inside the held-out tail.
   A posterior that concentrates fast is exactly what a disappearing expert
   punishes. This is the same non-stationarity `fixed_share.py` addresses,
   showing up as a concrete failure rather than a theoretical worry. See
   `markov_ogd.py` below for a follow-up that independently explains part of
   why the worst-case bound undershoots, via dependent-sample theory rather
   than a grid search.

**A live bug this correction exposed and fixed**: `hedge_scaled`/`ftrl_scaled`'s
internal `mult` is always relative to the UNSCALED textbook schedule, so the
sanity-check assertions in `main()` must compare against `mult=
HEDGE_STEP_MULTIPLIER` / `FTRL_STEP_MULTIPLIER` -- NOT `mult=1.0` -- to equal
the shipped algorithms. The assertions (and the `RATE_MULTIPLIERS`/
`HEDGE_FTRL_MULTIPLIERS` grids, plot titles, and the "best setting per family"
summary's baseline lookup) were all updated together; before this, running the
script after the `sleeping_experts.py` step-multiplier change would have
raised `AssertionError` at the first line of `main()`.

---

## `markov_ogd.py` — *extension*

Two DIFFERENT things both called "Markov Chain OGD" in the literature/casual
usage, both tested here after the user pointed at arxiv.org/abs/1809.04216
(Sun, Sun & Yin 2018, "On Markov Chain Gradient Descent") mid-session.

**Part A -- a Markov transition on the WEIGHT VECTOR.** After the usual OGD
gradient step and simplex projection, mass is moved by a row-stochastic
matrix `M` restricted to the awake subset. Fixed-Share is the special case
of a uniform leak; this generalises it using panel structure the uniform
leak throws away -- the 24 experts are 12 opening/closing PAIRS of the same
firm, and split into two PHASES. Three kernels tested (`build_structure`,
`transition`): `uniform` (Fixed-Share), `pair` (leak toward one's own
opening/closing counterpart), `phase` (leak toward same-phase experts).

| Function | What it does |
|---|---|
| `build_structure(panel)` | Per-expert counterpart index (B365<->B365C) and phase flag |
| `transition(kind, alpha, awake_idx, ...)` | Builds the row-stochastic matrix for one kernel, mass-preserving by construction |
| `run_markov_ogd(...)` | OGD + transition, under Sleeping Experts. `kind="identity"` asserted to exactly equal `run_ogd_sleeping` |

**Key finding: none of the three kernels help.** Every one selects alpha=0 on
the train prefix -- i.e. collapses to plain OGD, confirmed by ties with
mean_diff=0.000000 exactly (not approximately) against plain OGD. Consistent
with the Fixed-Share and joint-alpha/step findings elsewhere: once the step
itself is tuned, there is nothing left for post-hoc mass movement to fix,
because both attack the same "how fast does it forget" knob. Worth noting for
anyone re-deriving this: at LARGE alpha, the structured kernels (`pair`,
`phase`) are measurably less damaging than `uniform` -- so the panel structure
is real and detectable, it just isn't needed once the step is right.

**Part B -- MCGD in the sense of Sun, Sun & Yin: DEPENDENT SAMPLING.** This is
what that paper actually studies -- SGD where samples come from a Markov
chain trajectory instead of i.i.d. draws, not a transition on the weights.
Practical consequence (`effective_sample_schedule`): if the dependency
horizon is `tau` rounds, only ~`t/tau` of the first `t` samples are
effectively independent, so the step should decay on `t/tau` rather than
`t`, i.e. be inflated by roughly `sqrt(tau)` relative to the i.i.d. schedule.
Using `tau = T^(1/3) ~ 42` (the SAME block length `moving_block_bootstrap_test`
already uses, chosen independently for a different reason), this predicts an
inflation factor of `sqrt(42) * 13.82/3.76 ~ 23.8` on top of the honest-M
correction. That is a genuine, independent theoretical partial explanation
for the empirically-tuned x100 -- not the whole factor, but arrived at with
no grid search at all, which is why it's worth keeping in the write-up even
though it doesn't beat the tuned x100 outright (best found: tau=500,
mult=3.7, held-out 0.997648, essentially tied with shipped 0.997645).

**Outputs:** `results/markov_ogd.csv`, `_significance.csv`,
`results/markov_ogd.png`

---

## `mcmc_kelly.py` — *extension*

Does letting bet-sizing shrinkage EMERGE from posterior uncertainty beat the
arbitrary fixed 1/4-Kelly the pipeline ships? Models the outcome distribution
theta as `Dirichlet(1,1,1)` prior, with each awake expert's forecast treated
as `Dirichlet(kappa * theta)` evidence weighted by its learned OGD weight,
and maximises EXPECTED log growth under the resulting posterior rather than
plugging in the point estimate. Because `log(1-f) -> -inf` as `f -> 1`,
posterior mass on low theta automatically punishes large stakes -- shrinkage
without a hand-picked constant, and its size can vary round to round with how
much the experts actually agree.

No `scipy` dependency (not part of this project) -- carries its own
Lanczos-approximation `lgamma`, checked against `math.lgamma` on import.
Sampling is random-walk Metropolis-Hastings, run as `N_SAMPLES` PARALLEL
chains for `N_BURN` iterations rather than one long chain (`sample_posterior`)
-- turns a scalar Python loop into vectorised numpy steps, which is what
makes the per-bet cost tractable across thousands of bets.

| Function | What it does |
|---|---|
| `sample_posterior(p_experts, weights, kappa, rng)` | MH sampler, returns (n_chains, 3) posterior draws |
| `posterior_kelly(theta_draws, k, odds_k, mult)` | Stake maximising posterior-expected log growth, via a coarse grid (objective is concave) |
| `raw_mcmc_stakes(...)` | The expensive pass, run ONCE per (bookmaker, kappa) at mult=1; `apply_mult` then rescales for 1/4-Kelly WITHOUT resampling |
| `fit_with_weights(...)` | Own leave-one-out OGD fit (not `value_betting.fit_leave_out`) because the posterior needs the LEARNED weights, not just who's awake |

kappa is selected on the train prefix, subsampled (`KAPPA_SELECT_STRIDE=3`)
because the sampler dominates runtime and kappa is a coarse smoothing choice
that doesn't need per-round precision.

**Key finding: MCMC Kelly does not beat fixed 1/4-Kelly, and sometimes loses
to it significantly.** Paired on identical rounds, 4 targets (WHC, PSC, BWC,
B365C) x 2 stake levels (full, 1/4) = 8 comparisons: 6 ties, **2 significant
losses for MCMC** (PSC at both levels, p<0.001; BWC at full Kelly, p=0.028),
**zero significant wins**. On PSC specifically the posterior approach makes an
already-losing position WORSE, not better (full-Kelly bankroll collapses to
~1e-5 vs plug-in's 0.0008). The selected kappa is also unstable across
bookmakers (200, 200, 20, 20, no visible pattern) -- a sign the selection is
fitting noise. Filed alongside Fixed-Share and the per-expert step clock: a
theoretically more principled idea that the arbitrary baseline beats anyway.

**Runtime note**: the naive version (resampling for every kappa candidate AND
for both stake levels independently) took 50+ minutes and was killed by the
harness with no traceback -- looked exactly like a crash but wasn't one. The
current version reuses one sampler pass across both stake levels
(`apply_mult`) and subsamples the kappa search, bringing it to ~15-20 min.
If extending this file, do not resample per multiplier -- the posterior draws
don't depend on the stake multiplier, only the argmax over the grid does.

**SIGN WARNING** (shared with `value_betting*.py`): scores log-GROWTH per
bet, higher is better -- the opposite of every log-loss table in the project.

**Outputs:** `results/mcmc_kelly.csv`, `results/mcmc_kelly_significance.csv`,
`results/mcmc_kelly.png`

---

## `bandit_oco.py` — *extension*

Every other OCO stage in this project is FULL INFORMATION: after each match the
learner is handed the whole per-expert loss vector. This file asks what
survives when it sees only ONE number per round — the loss of the forecast it
published itself. That is the bandit model, and it is the natural feedback
model for a deployed forecaster who records only its own score.

| Function | What it does |
|---|---|
| `project_to_shrunk_simplex(v, xi)` | Projection onto `{w_i >= xi/n, sum w = 1}`. `xi=0` returns `project_to_simplex` UNCHANGED, which is what lets the full-information modes reproduce `run_ogd_sleeping` bit for bit |
| `tangent_probe(g)` | Unit vector in the simplex's tangent space `{sum u_i = 0}`; `None` when `n=1` so there is no direction to probe |
| `run_bandit_ogd(...)` | The four information models: `full_linear` (asserted identical to `run_ogd_sleeping`), `full_exact`, `one_point` (Flaxman/Kalai/McMahan 2005), `two_point` (Agarwal/Dekel/Xiao 2010). `batch` mini-batches the estimate |
| `check_estimator_unbiased(...)` | Asserts the two-point estimate averages to the TANGENTIAL true gradient (rel. error, cosine) over 200k vectorised draws |
| `estimator_alignment(...)` | Mean cosine of a SINGLE-draw estimate with the true gradient — the diagnostic that explains everything below |
| `sweep_estimator(...)` | Three-stage train-only selection of `(xi, eta)` with `_extend_if_endpoint` |

**THE TRAP THIS FILE EXPOSED, and the reason to read it before writing any
Sleeping-Experts comparison against the uniform average.** Under the Sleeping
Experts reduction the weight vector moves *even at `eta = 0`*: every time the
awake set changes, the persistent state is re-projected onto a
different-dimensional simplex, and that churn alone produces non-uniform
weights — measured **0.69 away from uniform in max norm** with zero learning.
So "beats the uniform average" is not evidence that anything was learned. The
file therefore carries an explicit `eta = 0` control, and the decomposition is:

| | projection churn | + bandit learning |
|---|---|---|
| two-point | +0.000500 (41.4%) | +0.000264 (21.9%) |
| one-point | +0.000501 (41.5%) | **−0.000000 (0.0%)** |

of a total uniform→OGD margin of 0.001206.

**Key findings.** (1) **One-point bandit OCO learns nothing on this data.**
Train selection drives its step to ~1e-14 and the entire region 1e-14..9e-11
ties at the same score — that is the no-learning plateau and the selector is
taking its boundary. The cause is quantified: single-draw alignment with the
true gradient is **+0.002**, so the signal is 0.2% of the estimate's magnitude
and the effective ratio over T rounds is ~0.55, still below one. The horizon is
too short for a one-point estimator on a 25-dimensional simplex.
(2) **Mini-batching does not rescue it** — swept over 4 batch sizes × 5 decades
of step, no combination yields an interior optimum; batching tolerates a larger
step without damage but never produces a gain.
(3) Two-point contributes a real 21.9%, but its shrink parameter is driven to
the lower bound, and as `delta -> 0` the two-point estimate converges to the
exact directional derivative, i.e. it **stops being bandit**. Read its number
as an upper bound, not as bandit performance.
(4) Side finding: full information with the **exact** gradient ties with the
shipped linearised OGD (+0.000002, p=0.934), so the linearisation costs nothing
measurable here.

**Outputs:** `results/bandit_oco.csv`, `_significance.csv`, `.png`

---

## `oco_kelly.py` — *extension*

Learns the Kelly **multiplier** online instead of fixing it at 1/4 —
`f_t = lambda_t * f*_t`, keeping the closed-form `f*` and learning only the
shrinkage. Written after `value_betting_online_kelly.py` (which learns the
stake `f` directly) lost 9 of 20 paired comparisons with zero wins.

| Function | What it does |
|---|---|
| `bet_sequence(p_hat, odds, y, mask)` | The bets that will be placed. Outcome selection does not depend on the multiplier, so every method here plays the IDENTICAL bets — which is what makes the paired per-bet test valid |
| `simulate_fixed(seq, lam, T)` | Constant multiplier; asserted in `main()` to reproduce `value_betting.simulate_kelly_betting` exactly at 0.25 |
| `simulate_learned(...)` | `algo="ons"` (Online Newton Step, exploits exp-concavity) or `"ogd"`. In 1-D the A-norm projection is ordinary clipping |

**Why the earlier version had to fail (the comparator argument).** A regret
bound for a stake learner is a promise against the best **constant** `f`. The
closed-form rule is not in that class: it produces a different `f` every round
from `p_hat` and the odds, both known *before* the bet. The learner also
discards `p_hat` — the quantity the two preceding OCO stages exist to produce —
and re-derives its stake from one bit per bet.

**Solvency, not tuning:** `lambda_max * max(f*) < 1` is a hard constraint —
above it a single loss takes the whole bankroll and the loss is unbounded — so
the admissible range is computed from the data, not chosen.

**Key findings.** Ties with fixed 1/4 (17 ties, 2 wins, 1 loss over
2 normalisations × 2 stages × 5 targets) and is ahead on final bankroll on the
profitable targets (WHC 6,494× vs 84×; BWC 7.1× vs 3.1×). The learned lambda
tracks target profitability — 0.37 on the loss-making PSC, 0.79 on WHC — which
a constant cannot express. **ONS selected a rate multiplier of exactly 1**, the
only place in this project where the theory's a-priori constant needed no
correction; OGD needed ×10 and hit a grid edge. Side finding: full Kelly ties in
16 of 20 cells, so the shipped 1/4 is more conservative than necessary on the
profitable targets, with its risk concentrated almost entirely on PSC.

**SIGN WARNING** (shared with `value_betting*.py`): log-GROWTH, higher is better.

**Outputs:** `results/oco_kelly.csv`, `_significance.csv`, `.png`

---

## `regime_switching.py` — *extension*

A Markov chain over latent **market states**, not over the weight vector.
`markov_ogd.py` already tried the latter and every kernel selected alpha=0.
Here K latent regimes each carry their own Sleeping-Experts OGD weight vector,
tied by the forward recursion of a hidden Markov model, with each regime
stepping in proportion to its responsibility. `K=1` reproduces
`run_ogd_sleeping` exactly for any setting of the knobs (asserted).

Motivated by a measured latent structure: `information_arrival.py` found the
closing advantage varies 26× with how much the line moved. The regime is left
**latent** so the file can ask whether the chain rediscovers that on its own.

| Function | What it does |
|---|---|
| `run_regime_ogd(..., beta, gamma, clock)` | `beta` = belief sharpness, `gamma` = forgetting, `clock` = per-regime step clock on accumulated responsibility. `beta=1, gamma=0` is exact Bayes, the original behaviour |
| `movement_per_match(match_ids)` | Own small copy of `information_arrival.py`'s line-movement statistic, used only as an after-the-fact DIAGNOSTIC, never inside the prediction |
| `most_variable_regime(B, mov, ok)` | Bins movement by belief quintile; returns `None` when no belief varies enough to stratify, which is the normal outcome when the chain concentrates |

**Key findings, and the first failure is the instructive part.** The first
version had only `tau`, and the mean belief sat at exactly **0.500** for both
regimes with zero correlation to anything: the regimes **never differentiate**,
because they see the same loss vector, so from a near-symmetric start they
follow identical trajectories. The result was simply OGD with its step divided
by K, and it lost significantly (+0.000128, p<0.001). The symmetric solution is
a fixed point of the dynamics and needs an explicit differentiation mechanism.

After adding beta/gamma/clock it **ties** with OGD (−0.000016, p=0.211), with
beta=30 a genuine interior optimum (grid extended to 1000). But two things
kill any claim that the chain works: the held-out surface is **flat to 2e-6
across three decades of beta** while the train differences driving the
selection are ten times larger, and **tau=0 is selected with belief sd =
0.0000** — the belief freezes early, so there is no Markov switching at all and
what remains is a fixed blend of K OGD runs, i.e. ensembling. The latent regime
does not correlate with measured line movement.

Third independent instance of the project's most robust pattern: once the step
is tuned, no way of moving mass over experts adds anything. The difference here
is that the failure was *diagnosed* rather than merely observed.

**Outputs:** `results/regime_switching.csv`, `_significance.csv`, `.png`

---

## `make_appendix_tables.py` — *support script, not part of the main pipeline*

Generates `docs/appendix_tables.tex`, which `docs/thesis.tex` `\input{}`s as
its results appendix. Reads every `results/*.csv` this project produces and
emits one LaTeX `longtable` per file, with Greek column headers (`HEAD` dict),
adaptive font size and column-dropping so wide tables still fit the text
block (`longtable()` -- drops the widest column and shrinks the font until
the estimated printed width fits, clips long free-text cells like verdict
strings to 20 characters).

**Run this after any pipeline rerun, before rebuilding `thesis.tex`** -- it is
not wired into the pipeline automatically, so the appendix goes stale
silently otherwise. Idempotent; writes nothing except the one `.tex` file.

**Outputs:** `docs/appendix_tables.tex` (not `results/`, note the different
directory -- this is a document-generation script, not an experiment)

---

## `temporal_trends.py` — *extension*

Is the market getting more reliable over time? Looks at log-loss, ECE and
overround per season, restricted to the 4 bookmakers present in all 10
seasons (B365, BW, PS, PSC) plus the 4 algorithms, so season-to-season
differences aren't confounded with coverage changes. Uses the
non-parametric Mann-Kendall test for trend significance instead of an OLS
slope, which suits a series of only 10 seasons.

| Function | What it does |
|---|---|
| `mann_kendall_test(x)` | Mann-Kendall trend test (S, Z, two-sided normal-approx p-value, direction) |
| `per_season_logloss(...)` / `per_season_ece(...)` | Season-level log-loss / ECE for one series |
| `per_season_overround(season_order)` | Mean overround per bookmaker per season, read straight from `odds_long.csv` (outcome-independent) |
| `plot_trends(...)` | Saves the 3-panel (log-loss / overround / ECE) trend plot |
| `main()` | Builds the per-series and aggregate tables, runs the trend tests, saves everything |

**Outputs:** `results/temporal_trends_table.csv`, `results/temporal_trends_mk_test.csv`,
`results/temporal_trends.png`

**Key finding:** no significant trend in log-loss or ECE (all Mann-Kendall
p > 0.37), but overround increases significantly (bookmaker average 4.42%
in 2016/17 to 6.0% in 2025/26, p = 0.0007) -- the opposite of a "maturing,
more competitive market" story.

---

## `pooling_window_comparison.py` — *extension*

Does the Tier-A ranking / significance story change with how much history
is pooled? Reruns the `final_ranking.py` pipeline and a key-significance
battery independently, from scratch, on a 5-year window (2021/22-2025/26)
and the full 10-year window, on the same 22-division panel, isolating the
effect of time depth from the later expansion in leagues.

| Function | What it does |
|---|---|
| `run_window(P, awake, y, seasons, panel, season_set)` | Refits every algorithm on one window, returns the ranking table, significance table and round count |
| `plot_comparison(ranking_table, out_path)` | Saves the 5yr-vs-10yr calibrated log-loss slope chart |
| `main()` | Runs both windows, saves both tables and the plot |

**Outputs:** `results/pooling_window_ranking_table.csv`,
`results/pooling_window_significance_table.csv`, `results/pooling_window_comparison.png`

---

## `process_data_shin.py` (deprecated)

Was a standalone variant of `process_data.py` that de-vigged with Shin's
(1992) method instead of proportional normalization -- the question it
answered ("is the project's basic de-vig choice load-bearing?") is now
answered by `data_processer.py`'s unified two-normalization output, so this
file moved to `scripts/deprecated/` alongside `process_data.py`. Its own
`shin_normalize` was a vectorized bisection (numpy broadcasting);
`data_processer.py`'s is a deliberately simpler, un-vectorized per-match
loop -- both verified to produce value-identical output.

**Key finding (still holds):** the fitted insider-trading proportion z
tracks overround almost exactly (lowest-margin bookmaker has the lowest z,
~0.5%; highest-margin has the highest, ~4.4%) -- a sensible internal
consistency check on the method itself.

---

## `market_devig_comparisons.py` — *extension/robustness check, consolidated*

Every "does market phase or de-vig method change the answer" experiment in
the project, as opposed to `value_betting.py`'s economic questions.
Consolidates 4 previously separate scripts (`opening_vs_closing.py`,
`shin_vs_basic_comparison.py`, `shin_vs_basic_calibration.py`, and
`opening_closing_quality_transfer.py`'s correlation half) into one file, now
in `scripts/deprecated/` -- see that folder's README for the exact mapping.

| Function | What it does |
|---|---|
| `split_by_market_phase(panel)` / `load_universe_from(csv_path)` | This file's own small copies (see `value_betting.py`'s equivalents for why each merged file carries its own rather than cross-importing) |
| `fit_all_algorithms(P, awake, y)` | Hedge/OGD/FTRL/Uniform average on one panel, returns `{name: phat}` -- replaces 3 copies of this exact weight-computation block that used to be scattered across the 4 merged scripts |
| `run_opening_vs_closing(panel, P, awake, y, label, suffix, with_ranking)` | Experiment 1: reruns the 4 algorithms on opening-only / closing-only / full panels, bootstrap-tests all 3 pairwise comparisons per algorithm, builds the combined ranking against every individual bookmaker. `main()` calls it TWICE -- once on the basic panel (`label="Basic"`, unsuffixed outputs, ranking on) and once on the Shin panel (`suffix="_shin"`, ranking off). The ranking is Basic-only on purpose: it reads the individual-bookmaker rows from `results_table.csv`, which is built from the basic dataset, so pairing them with Shin algorithm rows would mix two panels in one table |
| `run_quality_transfer_correlation(panel)` | Experiment 2: Pearson + Spearman correlation between a bookmaker's opening-side and closing-side log-loss, 13 pairs |
| `run_shin_vs_basic_comparison()` | Experiment 3: reruns the core algorithm + Tier-A-bookmaker comparison (log-loss/Brier, raw + calibrated) independently on both de-vig datasets |
| `run_shin_vs_basic_calibration()` | Experiment 4: raw-probability ECE/reliability comparison, both datasets |

**Outputs:**
`results/opening_vs_closing_table.csv`, `_significance.csv`, `.png`, `_ranking.csv`, `_ranking.png` (experiment 1, basic de-vig) |
`results/opening_vs_closing_table_shin.csv`, `_significance_shin.csv`, `_shin.png` (experiment 1 rerun on the Shin panel; no ranking, see above) |
`results/opening_vs_closing_vs_uniform.csv`, `_shin.csv` (each algorithm vs. the uniform-average baseline WITHIN each of the 3 panels, on all 3 metrics -- the "does learning pay off on a homogeneous panel?" test; free to compute, the per-round arrays already exist at that point in `run_opening_vs_closing`. Carries a `verdict` column naming the winner outright, because `mean_diff` is always algorithm-minus-uniform and accuracy's sign therefore reads the opposite way from log-loss/Brier) |
`results/opening_closing_quality_transfer.csv`, `.png` (experiment 2; the BW value-betting half of the original file is in `value_betting.run_bw_leakage_check` instead) |
`results/shin_vs_basic_table.csv`, `_significance.csv`, `_calibration_benefit.csv`, `.png` (experiment 3) |
`results/shin_vs_basic_calibration_table.csv`, `_ece.csv`, `_reliability.png` (experiment 4)

**Key finding:** closing odds significantly beat both opening odds AND the
full mixed 26-bookmaker panel, for all 4 algorithms -- opening odds
significantly hurt the mixture relative to the full panel too; in the
combined ranking, OGD (closing) is the best-ranked algorithm variant (#3
overall, behind only PSC and a single-season Betfair closing series).
**The same phase split rerun on the Shin panel reproduces this exactly**:
closing < full < opening, all 12 comparisons significant there too (24 of 24
significant across both normalizations), with Shin uniformly ~0.0005 lower
in every one of the 12 cells -- a level shift that changes no ordering, not
a different conclusion. So the "opening bookmakers are a net drag on the
mixture" result is not an artifact of proportional normalization.

**Second key finding (the vs-uniform test):** the split DOES change the
ranking -- not among the three OCO algorithms (OGD < Hedge < FTRL is
identical in all 6 panel x normalization cells) but in where the uniform
baseline lands: 4th on the full panel, 2nd on closing-only, 1st on
opening-only. Bootstrapped, that reversal is real: on the full panel all
three algorithms beat uniform significantly (6/6), while on the
phase-restricted panels **Hedge and FTRL lose significantly to uniform in
8 of 8 tests** and OGD ties (4/4 non-significant, p=0.36-0.55). Learning
only pays when the experts differ in quality; on a homogeneous panel the
adaptation cost isn't recovered. **Brier gives the identical verdict in all
18 cases; accuracy gives 0 significant results out of 18** (p=0.098-0.985),
so accuracy's apparently different rankings are noise. See WORK_PLAN.md for
the full reading. No
reliable evidence that closing-side skill transfers to opening-side skill
(13 pairs: Pearson r=-0.006; 7 multi-season pairs only: r=-0.390, Spearman
rho in [+0.18, +0.29] -- unstable sign, small sample). Shin vs. Basic:
ranking is IDENTICAL either way (PSC #1, then OGD>Hedge>FTRL>Uniform, then
the rest); Shin significantly improves raw log-loss/Brier everywhere
(p<0.0001, 11/11 entities), but the advantage mostly evaporates after
calibration (log-loss significant for only 4/11, Brier for 0/11);
calibration itself stops being significant for 9/11 series once Shin is
applied (vs. 11/11 under basic normalization) -- Shin and temperature
scaling correct the same favorite-longshot bias via different mechanisms,
so doing one makes the other largely redundant. Directly confirmed
mechanistically: ECE roughly halves under Shin for every series (-35% to
-57%, e.g. OGD: 0.0091 -> 0.0043), the calibration-gap curve visibly
flattens.

**Verified during the merge:** all 9 consolidated CSV outputs matched their
pre-merge scripts' output value-for-value (4 of 9 byte-identical outright;
the rest matched after sorting, differing only in row order from ties /
dict-iteration order, not from any computational difference).

---

## Results files at a glance

| File | Produced by |
|---|---|
| `results/results_table.csv`, `results/sleeping_experts_comparison.png` | `sleeping_experts_experiment.py` |
| `results/calibration_table.csv`, `results/calibration_reliability.png` | `calibration_analysis.py` |
| `results/calibration_correction_table.csv`, `results/calibration_correction.png` | `calibration_correction.py` |
| `results/final_ranking_table.csv`, `results/final_ranking.png` | `final_ranking.py` |
| `results/significance_test_table.csv` | `significance_test.py` |
| `results/value_betting_table.csv`, `.png` | `value_betting.run_original_two` |
| `results/fixed_share_table.csv`, `results/fixed_share_weights.csv` | `fixed_share.py` |
| `results/contextual_experts_table.csv`, `results/contextual_experts.png` | `contextual_experts.py` |
| `results/calibration_per_season_table.csv`, `results/calibration_per_season_significance.csv`, `results/calibration_per_season.png` | `calibration_per_season.py` |
| `results/calibration_warmstart_table.csv`, `results/calibration_warmstart_significance.csv`, `results/calibration_warmstart.png` | `calibration_warmstart.py` |
| `results/opening_vs_closing_table.csv`, `_significance.csv`, `.png`, `_ranking.csv`, `_ranking.png` | `market_devig_comparisons.run_opening_vs_closing` (basic de-vig) |
| `results/opening_vs_closing_table_shin.csv`, `_significance_shin.csv`, `_shin.png` | `market_devig_comparisons.run_opening_vs_closing` (Shin de-vig) |
| `results/outcome_class_accuracy_table.csv`, `results/outcome_class_accuracy.png` | `outcome_class_accuracy.py` |
| `results/accuracy_by_band_reliability.csv`, `_confidence.csv`, `_significance.csv`, `results/accuracy_by_band.png` | `accuracy_by_band.py` |
| `results/value_betting_by_band_table.csv`, `_holdout.csv`, `results/value_betting_by_band.png` | `value_betting_by_band.py` |
| `results/outcome_class_comparison_descriptive.csv`, `_significance.csv`, `results/outcome_class_comparison.png` | `outcome_class_comparison.py` |
| `results/outcome_class_accuracy_recall_tests.csv`, `_precision_ci.csv`, `results/outcome_class_accuracy_test.png` | `outcome_class_accuracy_test.py` |
| `results/value_betting_circular_table.csv`, `_vs_honest.csv`, `results/value_betting_circular.png` | `value_betting_circular.py` |
| `results/value_betting_sharp_anchor_table.csv`, `_significance.csv`, `results/value_betting_sharp_anchor.png` | `value_betting_sharp_anchor.py` |
| `results/learning_rate_study.csv`, `results/learning_rate_study.png` | `learning_rate_study.py` |
| `results/no_pinnacle_ranking.csv`, `_significance.csv`, `results/no_pinnacle.png`, `results/no_pinnacle_closing.png` | `no_pinnacle_panel.py` |
| `results/best_configuration.csv`, `_significance.csv`, `results/best_configuration.png` | `best_configuration.py` |
| `results/deployment_test.csv`, `_significance.csv`, `results/deployment_test.png` | `deployment_test.py` |
| `results/recent_window_ranking.csv`, `_significance.csv`, `_panels.csv`, `_configs.csv`, `_no_pinnacle.csv`, `_live_market.csv`, `results/recent_window.png` | `recent_window.py` |
| `results/recent_window_value_betting.csv` | `recent_window_value_betting.py` |
| `results/information_arrival_strata.csv`, `_significance.csv`, `results/information_arrival.png` | `information_arrival.py` |
| `results/temporal_trends_table.csv`, `results/temporal_trends_mk_test.csv`, `results/temporal_trends.png` | `temporal_trends.py` |
| `results/pooling_window_ranking_table.csv`, `results/pooling_window_significance_table.csv`, `results/pooling_window_comparison.png` | `pooling_window_comparison.py` |
| `results/opening_closing_quality_transfer.csv`, `.png` | `market_devig_comparisons.run_quality_transfer_correlation` |
| `results/opening_value_betting_corrected.csv`, `.png` | `value_betting.run_bw_leakage_check` |
| `results/value_betting_all_bookmakers_table.csv`, `.png` | `value_betting.run_all_bookmakers_sweep` |
| `results/value_betting_training_panel_sweep.csv` | `value_betting.run_training_panel_sweep` |
| `results/value_betting_closing_trained_table.csv`, `.png` | `value_betting.run_closing_trained_both_norms` |
| `results/value_betting_online_kelly_table.csv`, `_significance.csv`, `.png` | `value_betting_online_kelly.py` |
| `results/value_betting_ogd_vs_ftrl_table.csv`, `_significance.csv`, `.png` | `value_betting_ogd_vs_ftrl.py` |
| `results/bandit_oco.csv`, `_significance.csv`, `.png` | `bandit_oco.py` |
| `results/oco_kelly.csv`, `_significance.csv`, `.png` | `oco_kelly.py` |
| `results/regime_switching.csv`, `_significance.csv`, `.png` | `regime_switching.py` |
| `data/processed/odds_long.csv`, `data/processed/odds_long_shin.csv` | `data_processer.py` |
| `results/shin_vs_basic_table.csv`, `_significance.csv`, `_calibration_benefit.csv`, `.png` | `market_devig_comparisons.run_shin_vs_basic_comparison` |
| `results/shin_vs_basic_calibration_table.csv`, `_ece.csv`, `_reliability.png` | `market_devig_comparisons.run_shin_vs_basic_calibration` |
| `results/shin_vs_basic_value_betting_table.csv`, `_significance.csv`, `.png` | `value_betting.run_shin_vs_basic` |
| `docs/thesis_report.tex` / `.pdf` | written by hand (XeLaTeX) |
| `docs/supervisor_report.tex` / `.pdf` | written by hand (XeLaTeX) -- ~9pp "what has been built" digest; `docs/progress_report.tex` is its superseded predecessor |
| `docs/status_report.tex` / `.pdf` | written by hand (XeLaTeX) -- ~5pp, the shortest write-up, findings and numbers only, deliberately no theory (see WORK_PLAN.md) |
