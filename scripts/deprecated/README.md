# Deprecated

Superseded by `../data_processer.py`, which produces BOTH processed
datasets (`odds_long.csv` via proportional normalization, `odds_long_shin.csv`
via Shin normalization) from one run instead of two separate scripts.

- `process_data.py` -- produced `odds_long.csv` only.
- `process_data_shin.py` -- produced `odds_long_shin.csv` only; imported
  shared constants/helpers from `process_data.py`, which is why the two are
  kept together here.

Verified byte-identical (and value-identical, for the Shin file) output
against `data_processer.py` before being moved here. Kept for reference
only -- nothing else in the project imports from either file.

## Value-betting family -> `../value_betting.py`

Superseded by `../value_betting.py`, which consolidates every value-betting
experiment (except online-Kelly, see below) into one file with a single
generalized leave-one-out fitting routine (`fit_leave_out`) replacing what
used to be five near-identical copies of the same "restrict panel, fit OGD,
calibrate" logic.

- `value_betting_all_bookmakers.py` -> `run_all_bookmakers_sweep`
- `value_betting_closing_trained.py` -> `run_closing_trained_both_norms`
- `value_betting_training_panel_sweep.py` -> `run_training_panel_sweep`
- `shin_vs_basic_value_betting.py` -> `run_shin_vs_basic`
- `opening_closing_quality_transfer.py` -- split in two: its value-betting
  half (`run_opening_panel_bw_test`) -> `../value_betting.py`'s
  `run_bw_leakage_check`; its correlation half (`build_pair_table`,
  `correlate`, `plot_quality_transfer`) -> `../market_devig_comparisons.py`'s
  `run_quality_transfer_correlation`.

The original two-bookmaker test (`main()` in what's now `../value_betting.py`
itself) became `run_original_two`.

`../value_betting_online_kelly.py` (the projected-OGD stake learner) was
deliberately NOT merged in -- it swaps out the staking mechanism itself
rather than varying the training panel or normalization, so it stays a
separate file, now importing its shared helpers
(`split_by_market_phase`, `select_bookmakers`, `load_universe_from`) from
the consolidated `../value_betting.py` instead of these retired files.

Every consolidated CSV output was verified value-identical (`pd.testing.
assert_frame_equal` after sorting, tolerant of row order and floating-point
noise) against these scripts' pre-merge output before they were moved here.
One subtlety caught by that verification: the "mixed-trained" / full-panel-
minus-target fits (`test_closing_bookmaker` here) never applied the
">=1 of the training panel awake" round-reduction that the opening-only/
closing-only fits (`test_opening_bookmaker`, `test_closing_trained`) did --
`fit_leave_out`'s `apply_reduction` parameter preserves that distinction.

## Market-structure / de-vig comparisons -> `../market_devig_comparisons.py`

Superseded by `../market_devig_comparisons.py`, which consolidates the
remaining "does market phase / de-vig method change the answer" experiments
into one file, with a shared `fit_all_algorithms` helper replacing three
copies of the same Hedge/OGD/FTRL/Uniform-average weight computation.

- `opening_vs_closing.py` -> `run_opening_vs_closing`
- `shin_vs_basic_comparison.py` -> `run_shin_vs_basic_comparison`
- `shin_vs_basic_calibration.py` -> `run_shin_vs_basic_calibration`
- `opening_closing_quality_transfer.py`'s correlation half -> see above

Every consolidated CSV output was verified value-identical against these
scripts' pre-merge output (4 of 9 byte-identical outright; the rest matched
after sorting, differing only in row order from ties/dict-iteration order)
before they were moved here.
