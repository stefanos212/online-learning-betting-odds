# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A thesis project ("Online Convex Optimization based tuning of Mixture of Experts for Sports Outcome
Prediction"): treats each football betting site as an "expert" giving H/D/A probabilities per match, and
learns online (Hedge / OGD / FTRL) how to weight/combine them, evaluated against the raw bookmakers
themselves. Not a conventional software repo -- no git, no build step, no linter, no test suite. It's a
sequential pipeline of standalone Python scripts under `scripts/`, each reading/writing flat CSV/PNG files
under `data/` and `results/`, plus a LaTeX write-up under `docs/`.

**Provenance**: this is a cleaned-up copy of the earlier exploratory project `thesis_sports_prediction`
(still on disk next to this folder, untouched -- nothing was deleted there). Only thesis-relevant pieces
were carried over. Deliberately left behind: `online_experts.py` (superseded 6-bookmaker first draft),
`generate_report.py` + `thesis_progress_report.pdf` (old reportlab report, replaced by the LaTeX report),
LaTeX aux files, and the old project's `.claude/` local permission settings. Pull anything back from there
if it turns out to be needed.

**Read `docs/WORK_PLAN.md` first** for a narrative of what's been done, key findings so far (several of
which reverse earlier, smaller-dataset findings), and open next steps. Read `docs/CODE_OVERVIEW.md` for a
function-by-function index of every script. Don't duplicate either here -- keep this file to orientation +
non-obvious gotchas.

**Language**: the user works in Greek and the thesis write-up (`docs/thesis.tex`) is in Greek.
Reply in Greek and write user-facing thesis documents in Greek; code, file names, docstrings and
`docs/*.md` stay in English, and technical terms (Hedge, OGD, log-loss, ...) stay in English inside Greek
prose.

## Running things

- No venv; dependencies (pandas, numpy, matplotlib) are plain `pip install`s into the system/user Python.
  `pymupdf` is only needed to render a compiled PDF to PNG for visual review. No build/lint/test commands
  exist for the Python side.
- Environment is Windows + PowerShell (not bash/WSL) -- use PowerShell syntax.
- Run any script from inside `scripts/`: `cd scripts; python <name>.py` -- the inter-script imports (e.g.
  `from sleeping_experts import ...`) resolve relative to that directory, not the repo root. Data/results
  paths in most scripts are hardcoded relative to `scripts/` (e.g. `../data/processed`, `../results`), so
  they only resolve correctly when run from inside `scripts/` -- this replaced an earlier `__file__`-based
  `ROOT` computation that worked from any cwd (`download_data.ps1` still uses that older pattern).
- Nothing is cached between scripts except via the CSVs each one writes to `results/` and later scripts
  read back -- every script recomputes its algorithms from scratch on every run. Exception worth knowing:
  `market_devig_comparisons.py` reads `results/results_table.csv` (written by
  `sleeping_experts_experiment.py`, not `sleeping_experts.py` itself -- see the next bullet) for the
  individual-bookmaker rows of its combined ranking and its opening/closing correlation check.
- **`sleeping_experts.py` is algorithms/data-loading only** (`load_full_universe`, the 3
  `run_*_sleeping` algorithms, `last_awake_weight`, `evaluate`, the shared constants) -- it has no
  `main()` and produces no output; every other active script imports from it, directly or not.
  `sleeping_experts_experiment.py` is the script that actually runs those algorithms over the full panel
  and writes `results_table.csv` + `sleeping_experts_comparison.png`.
- **Three step multipliers in `sleeping_experts.py` are load-bearing**: `OGD_STEP_MULTIPLIER = 100`,
  `HEDGE_STEP_MULTIPLIER = 0.25`, `FTRL_STEP_MULTIPLIER = 0.05`. All three are empirical, selected on a
  chronological train prefix (same protocol as the calibration learning rate). Touching any of them
  moves every number in `results/` and in the write-ups.
  **The lesson that produced two of them: a grid's endpoint is not an optimum.** `learning_rate_study.py`
  originally swept 1, 2, 5, ... 1000 and reported Hedge and FTRL as "already optimal at 1.0". They were
  at the grid's LOWER BOUNDARY and nobody had looked left. Searching downwards found genuine interior
  minima (Hedge x0.25, FTRL x0.05; +0.00013 and +0.00018 held-out). Whenever a sweep's answer sits on an
  endpoint, extend the grid before believing it.
  The directions differ for a structural reason: OGD's eta multiplies one round's gradient, so raising it
  moves further; Hedge's sits in an exponent over one round's loss, so lowering it lengthens memory.
  **NOT stale on the 26-expert panel — this was checked and the earlier "stale" note was wrong.**
  `learning_rate_study.py` prints its winner as "best setting per family, chosen by HELD-OUT tail",
  and by that criterion the 26-expert optima are x0.05 (Hedge) and x0.005 (FTRL), which is where the
  claim that the shipped constants had gone stale came from. But **selecting on the tail is selecting
  on the test set** — the protocol in the methodology chapter selects on the chronological 75% train
  prefix. log-loss is a plain mean over rounds, so the train figure is recoverable exactly from the
  full-sample and tail columns, `(full*T - tail*n_val)/n_train` (verified against a direct evaluation
  on the train mask to 1e-16). Under the protocol the answer is:
  **Hedge train-picks x0.25 and FTRL train-picks x0.05 — exactly the shipped values**; OGD train-picks
  x110 against the shipped x100, with an identical tail of 0.99765, i.e. a tie inside the documented
  x75–x110 plateau. The tail-selected alternatives are *worse on train* and the ordering reverses
  (Hedge x0.05: train 0.99838 vs x0.25's 0.99828; FTRL x0.005: train 0.99862 vs x0.05's 0.99828),
  which is the signature of fitting the tail. **So there is nothing to re-tune and no pipeline rerun
  is owed.** When reading that script's summary, derive the train column before believing its winner.
- **Six optimisation ideas were tried and REJECTED** (don't re-derive them). (1) Fixed-Share alpha swept
  jointly with the Hedge step: alpha>0 loses at every step. alpha only ever helped because the step was
  too large -- they are substitutes for the same quantity, "how fast it forgets", and the step is the
  better instrument. (2) Step and calibration rate swept jointly instead of in two stages: no interaction
  at all, the best calibration rate is 0.15 in 11 of 11 rows, and joint selection on train picks x150 for
  OGD which is WORSE held-out (flat plateau -> train selection picks noise). (3) A per-expert step clock
  (n_i awake rounds instead of global t, which is what the Sleeping Experts regret bound is actually
  about): better on train, **-0.00073 held-out** for OGD -- rare experts get ~14x larger steps and the
  mixture chases single-season bookmakers. Textbook overfitting, and the clearest instance of it in the
  project. (4) AdaGrad, per-expert and scalar: recovers 20% at best. (5) `markov_ogd.py`: a Markov
  transition kernel on the weight vector (uniform/pair/phase leaks) -- every kernel selects alpha=0 on
  train, ties with plain OGD exactly. (6) `mcmc_kelly.py`: Bayesian/MCMC posterior-expected Kelly sizing
  instead of fixed 1/4 -- 6 of 8 paired tests are ties, 2 are SIGNIFICANT LOSSES (PSC both stake levels,
  BWC at full Kelly), zero significant wins. Still open and untried in the right range: a CONSTANT
  (non-decaying) step -- the old sweep only tried constants >= 0.1, which we now know is far too large.
  Also see `markov_ogd.py`'s SECOND result (not a rejection): MCGD in the dependent-sampling sense (Sun,
  Sun & Yin 2018) independently predicts a step inflation of ~24x from theory alone (no grid search),
  against the empirically-tuned 100x -- a partial theoretical explanation worth keeping in the write-up.
  **Three more were added later.** (7) `bandit_oco.py`: bandit OCO with one- and two-point gradient
  estimators. The one-point estimator learns NOTHING here -- train selection drives its step to ~1e-14
  and the whole region 1e-14..9e-11 ties, because its per-round alignment with the true gradient is
  +0.002, so the effective signal-to-noise over T rounds is ~0.55, below one. Mini-batching (the one
  fix the variance argument licenses) was swept over 4 batch sizes x 5 decades of step and never
  produces an interior optimum. **The trap this file exposed, and the reason to read it before any
  Sleeping-Experts comparison**: the weight vector moves even at eta=0, because the projection is
  re-done onto a different-dimensional simplex every time the awake set changes -- measured 0.69 away
  from uniform in max norm. So "beats the uniform average" is NOT evidence of learning; 41% of the
  mixture's whole margin over uniform is that projection churn. Always include an eta=0 control.
  (8) `regime_switching.py`: a Markov chain over latent MARKET states rather than over the weight
  vector. First version failed with the mean belief pinned at exactly 0.500 -- the regimes never
  differentiate, because they see the same loss vector, so it was just OGD with the step divided by K.
  After adding a belief sharpness, forgetting, and a per-regime clock it ties with OGD (-0.000016,
  p=0.211), but tau=0 is selected and the belief's sd is 0.0000, so the Markov part provably does
  nothing and what remains is ensembling. The latent regime does not correlate with the measured
  line-movement structure at all. (9) `oco_kelly.py` is the one NON-rejection of the batch -- see below.
- **`oco_kelly.py` -- the one idea in this family that did NOT fail, and why.** The earlier
  `value_betting_online_kelly.py` learned the stake fraction `f` directly and lost 9 of 20 paired
  comparisons with zero wins. The diagnosis is the COMPARATOR: a regret bound for that learner is a
  promise against the best CONSTANT f, and the closed-form Kelly rule is not in that class at all,
  since it produces a different f every round from `p_hat` and the odds -- both known before the bet.
  It also throws away `p_hat`, the very thing the first two OCO stages exist to produce. Learning the
  **multiplier** instead (`f_t = lambda_t * f*_t`) keeps the round-specific information and makes the
  comparator meaningful, because "best constant lambda" is exactly the quantity `KELLY_MULT = 0.25`
  pins down. That version ties with fixed 1/4 (17 ties, 2 wins, 1 loss) and is ahead on final bankroll
  on the profitable targets. Two things worth remembering: the loss is exp-concave in lambda, so
  **Online Newton Step** applies and gets O(log T) where OGD gets O(sqrt(T)) -- and ONS selected a rate
  multiplier of exactly **1**, the only place in this project where the theory's a-priori constant
  needed no correction. Also, `lambda_max * max(f*) < 1` is a SOLVENCY constraint, not a tuning knob:
  above it one loss takes the whole bankroll and the loss stops being bounded, so the admissible range
  is read off the data.
- Full pipeline, in dependency order (only `data_processer.py` needs re-running if the raw data changes;
  everything else always re-derives its own numbers):
  `download_data.ps1 -> data_processer.py -> sleeping_experts.py -> sleeping_experts_experiment.py ->
  calibration_analysis.py -> calibration_correction.py -> final_ranking.py -> {significance_test,
  fixed_share, value_betting -> {value_betting_online_kelly, value_betting_ogd_vs_ftrl,
  value_betting_circular, value_betting_sharp_anchor, value_betting_by_band},
  contextual_experts, outcome_class_accuracy -> {outcome_class_comparison,
  outcome_class_accuracy_test}, temporal_trends, pooling_window_comparison,
  calibration_per_season -> calibration_warmstart, market_devig_comparisons -> {information_arrival,
  best_configuration -> {no_pinnacle_panel, recent_window -> recent_window_value_betting}},
  deployment_test, accuracy_by_band, learning_rate_study, markov_ogd, mcmc_kelly, bandit_oco,
  oco_kelly, regime_switching}` -- see
  `docs/WORK_PLAN.md` and `docs/CODE_OVERVIEW.md` for what each covers. Three of these read another
  script's CSV rather than only the processed data, so run order matters for them:
  `market_devig_comparisons.py` and `value_betting.py` read `results_table.csv`;
  `value_betting_ogd_vs_ftrl.py` and `value_betting_circular.py` read
  `value_betting_all_bookmakers_table.csv` (the first of the two ASSERTS its own OGD half reproduces it,
  so a stale file there turns into a confusing failure).
  `no_pinnacle_panel.py` is deliberately standalone -- it drops PS/PSC from the panel and refits, and
  must NOT be wired into the main pipeline. The main pipeline's panel is 24 experts (see
  `EXCLUDED_BOOKMAKERS` below), not 26. It has no code dependency on `best_configuration.py`, but its
  second panel exists because of that script's finding, so read that one first.
  After any pipeline rerun, also rerun `scripts/make_appendix_tables.py` before rebuilding `thesis.tex`
  -- it is not wired into the dependency chain above and its output goes stale silently otherwise.
- **`EXCLUDED_BOOKMAKERS = ()` in `sleeping_experts.py` — EMPTY on purpose.** The panel is **26 experts
  and 76,584 rounds**. Interwetten (IW, IWC) used to be listed there and was **re-included** after the
  exclusion was re-examined; the constant is kept only as a hook. **Four files build a panel and all
  four still apply the filter**: `sleeping_experts.load_full_universe`, the `load_universe_from` copies
  in `value_betting.py` and `market_devig_comparisons.py`, and `contextual_experts.load_match_order`.
  Why it was reversed, so it is not re-derived: (1) the rule was **not symmetric** — the panel keeps 13
  series with 0% coverage in the first eight seasons (BFE/BFEC ~19%, 1XB/BF/BFC/1XBC ~10%, and seven
  more at 7-10%), the exact mirror image of IW's pathology, while IW is the SEVENTH best-covered series
  of the 26 and was the only one removed; (2) it treated the symptom — the real defect was a coverage
  floor computed over the whole decade plus a global intersection across compared series, and both have
  since been fixed where they mattered (`recent_window.py` recomputes its floor INSIDE its own window,
  so IW fails it there at 0% on its own merits, and `best_configuration.py` intersects only over the
  series it actually compares, which never included IW). See the module docstring for the full argument.
  Effects of re-inclusion worth knowing: OGD raw 0.99739 -> 0.99745, calibrated 0.99695 -> 0.99693; IW
  enters Tier A as the WORST-ranked bookmaker (raw 1.00186) which makes "we beat everyone but PSC"
  slightly cheaper; and it added a second significant value-betting profit (IW, 7.89x, p=0.008).
- **Copy drift was the recurring failure mode of the "every script self-contained" design, and the
  three loaders that caused it have now been consolidated.** `load_universe_from`,
  `load_match_order` and `split_by_market_phase` used to exist as identical copies in
  `value_betting.py`, `market_devig_comparisons.py` and `contextual_experts.py`; they now live once
  in `sleeping_experts.py` and the copies are gone. Downstream files that do
  `from value_betting import load_match_order` still work, because the name is re-exported through
  that module's own import. **Only 3 of the 9 duplicated function names were real duplicates** --
  `plot_comparison` (x5), `score` (x3), `_style_axes`, `band_labels`, `movement_per_match` and
  `_extend_if_endpoint` share a name but do different jobs, so merging them would change behaviour.
  Check before deduplicating by name.
  `fixed_share.py` still keeps its own Hedge recursion (it once missed `HEDGE_STEP_MULTIPLIER`),
  caught by the assertion it carries (`alpha=0 should exactly match run_hedge_sleeping()`) -- which is
  the argument for writing that kind of assertion into any new copy. After changing anything in
  `sleeping_experts.py`, grep for local re-implementations.
- **`load_universe_from` is cached to `data/cache/*.npz`**, keyed on the source CSV's path, size and
  mtime, so regenerating or editing `odds_long.csv` invalidates it automatically and there is no
  clearing step to forget. A cache hit costs 0.05s against 2.5s for the parse. Two traps found while
  building it: `np.savez` appends `.npz` to a filename that lacks it (so the temp file for the
  atomic rename must be opened as a handle), and `seasons`/`match_id` are object arrays that npz
  cannot reload under `allow_pickle=False`, so they are stored as unicode and cast back to object on
  read -- callers must see exactly what the uncached path returned. The round-trip test in
  `scratchpad` compares **every** field; an earlier version compared only the six that
  `load_universe_from` returns, missed `match_id`, and the dtype fault reached the pipeline.
- **`run_pipeline.py` runs the whole chain in dependency order** (`--list`, `--only`, `--from`,
  `--skip`, `--keep-going`, per-step timing). `data_processer` is opt-in via `--with-data`;
  `make_appendix_tables` runs last in both languages, since it is not a dependency of anything and
  goes stale silently. A full run is **~84 minutes**, dominated by genuinely different computations
  rather than repetition: mcmc_kelly 747s, value_betting 653s, bandit_oco 489s, oco_kelly 377s,
  regime_switching 302s, markov_ogd 262s, and the remaining 28 steps 2210s together. For scale,
  `load_full_universe` costs 2.4s x 22 callers = ~1% of the total, so caching it is a deduplication
  win and not a speed one.
- **`check_results.py` is the verification ritual, automated.** `save BASELINE` snapshots `results/`,
  `diff BASELINE` compares: CSVs on values (rtol 1e-9, sorted on a stable key, because row order can
  legitimately differ) and everything else by bytes. The pipeline is deterministic -- all five
  scripts that use randomness seed explicitly -- so a clean run reproduces `results/` exactly, which
  is what makes the comparison meaningful. Verified: 124/124 files identical.
- **Comparing series with different coverage: score them on ONE common round set, always.** A log-loss
  averaged over a different set of matches is not a comparable number, and this has produced two wrong
  conclusions in this project already (a near-tie between OGD and PSC in the ranking table, and a
  spurious win over B365C in the first no-Pinnacle-on-closing run). `best_configuration.py` and
  `no_pinnacle_panel.py` both do it the right way: intersect where every compared series is defined,
  fit on each series' full support but score only on the intersection, and add a chronological 75/25
  split because choosing a configuration by reading these numbers is itself selection. Corollary: never
  subtract a number computed on one round set from one computed on another -- "Pinnacle contributes
  0.99693 -> 0.99793" was exactly that mistake and is not measuring what it claims.
- **`scripts/deprecated/`**: superseded scripts, kept for reference only, not part of the active
  pipeline -- see `scripts/deprecated/README.md` for what replaced each one. Currently: `process_data.py`
  + `process_data_shin.py` (-> `data_processer.py`), and the 8 value-betting/market-comparison scripts
  consolidated into `value_betting.py` and `market_devig_comparisons.py` (opening/closing odds, Shin vs.
  basic normalization, and their value-betting variants -- each output verified value-identical against
  the pre-merge scripts before being moved here).
- **`docs/thesis_en.tex` is the ENGLISH edition of the thesis** (85pp), a full translation of
  `thesis.tex` that keeps exactly ONE Greek section (the Greek abstract, wrapped in
  `\begin{greek}...\end{greek}`; the preamble is `\setmainlanguage{english}` +
  `\setotherlanguage{greek}`). It is a SEPARATE file, not generated — **a result change has to be
  carried into `thesis.tex` AND `thesis_en.tex` by hand**, same as `status_report.tex`. Two
  conversion traps: the Greek edition writes numbers as `0{,}99745` / `76.584` and the English one
  must use `0.99745` / `76,584`; and `\newcommand{\en}[1]{#1}` is kept as an identity so stray
  `\en{}` from the Greek source is harmless. Its appendix is `appendix_tables_en.tex`, generated by
  `python make_appendix_tables.py --en` (the no-flag run still writes the Greek
  `appendix_tables.tex` byte-identically — verified) — **rerun BOTH after any pipeline rerun**.
  Both covers carry the TUC emblem from `docs/tuc_logo.png` (copy of `images.png`) via
  `\graphicspath{{../results/}{./}}`, and the real committee (Spyropoulos/supervisor, Lagoudakis,
  Samoladas). The Greek title needs `\Large`, not `\LARGE`, or it rewraps badly.
- **TWO active LaTeX write-ups in `docs/`, both Greek, both XeLaTeX**, quoting numbers from `results/`:
  `thesis.tex` (the full thesis, ~60pp -- title/abstract EL+EN, related work, theory with proofs, data,
  methodology, implementation, results with figures, discussion, conclusions, and an auto-generated
  appendix) and `status_report.tex` (~6pp supervisor status update, no theory). `thesis_report.tex`,
  `supervisor_report.tex` and `progress_report.tex` are superseded and now live in `docs/superseded/` --
  don't copy numbers out of them without checking `results/` first. **When a result changes, `thesis.tex`
  and `status_report.tex` both need updating by hand** (the thesis appendix regenerates via
  `make_appendix_tables.py`; everything else in both files does not).
- **`status_report.tex` carries NO theory, by explicit instruction** -- no de-vig / log-loss /
  temperature-scaling / Kelly formulas, no explanation of what Sleeping Experts is. The supervisor knows
  the theory; the report is what was done, what came out, and what it means. The student writes and
  edits this one directly and has asked three times for less method exposition, so if you touch it,
  prefer small targeted edits, keep their wording, and do not reintroduce derivations. The full
  treatment belongs in `thesis.tex`.
- **LaTeX build**: needs **XeLaTeX** (fontspec + polyglossia for Greek) -- pdfLaTeX will not work.
  MiKTeX is installed per-user at
  `C:\Users\Stefanos\AppData\Local\Programs\MiKTeX\miktex\bin\x64\`, which may not be on PATH in Claude's
  shell, so call `xelatex.exe` by full path, from inside `docs/` (`\graphicspath{{../results/}}` is
  relative to the working directory), 3 passes so refs/TOC settle. For test builds pass
  `-output-directory` pointing at a scratch dir so aux/log files don't land in `docs/` -- but note that
  **cross-references will NOT resolve that way** if a stale `.aux` is sitting in `docs/`: MiKTeX writes
  the new aux to the scratch dir and keeps reading the old one, so `\ref` stays undefined on every pass
  no matter how many you run. For a document with internal refs, either build in place or clear the
  stale aux first. The MiKTeX
  packages `libertine` and `dejavu` are required (fonts "Linux Libertine O" / "DejaVu Sans Mono").
  `.vscode/settings.json` defines a 3-pass xelatex recipe for the LaTeX Workshop extension (the default
  recipe uses latexmk, which needs Perl). After a build, check the `.log` for `Overfull`, `undefined`,
  and `Missing character` before calling it done.
- To visually review a compiled PDF: poppler isn't installed, so the Read tool's built-in PDF path won't
  work. Render pages to PNG with `pymupdf` (`import pymupdf; doc = pymupdf.open(path);
  doc[i].get_pixmap(dpi=70).save(...)`) and view those instead.
- **Never round-trip a UTF-8 file through `Get-Content`/`Set-Content`** (Windows PowerShell 5.1): it
  reads a BOM-less UTF-8 file as Windows-1252, so `(Get-Content f.tex -Raw) -replace ... | Set-Content
  f.tex -Encoding utf8` silently double-encodes every Greek character and corrupts the file. This
  happened to `docs/status_report.tex` and it had to be rewritten from scratch. Use the Edit tool for
  text substitutions in `.tex`/`.md`/`.py` files, never a shell round-trip.
- **DELETE `thesis.aux` (plus `.toc`, `.out`) BEFORE every build.** Observed repeatedly: after editing
  `thesis.tex`, the first xelatex pass writes an `.aux` containing NULL BYTES, and passes 2-3 then die
  with ~100 copies of `! Text line contains an invalid character` pointing at
  `\@writefile{toc}{\content^^@^^@...`. The `.tex` itself is fine -- the giveaway is that the error
  cites a line in the AUX, not in the source, and that pass 1 produced a PDF while 2 and 3 did not
  (count the `miktex-dvipdfmx` lines: three successful passes print three). `Remove-Item
  thesis.aux,thesis.toc,thesis.out` then rebuild, and it comes out clean every time. Don't go looking
  for the bug in the LaTeX.
- **A mojibake `.tex` HANGS xelatex silently** -- no error, no log output, not even under
  `-interaction=nonstopmode`; the process just sits there until killed (`taskkill /F /IM xelatex.exe`,
  and check `Get-Process xelatex` afterwards, a second one can survive). So if a build that worked
  before suddenly hangs, suspect the FILE (did something just rewrite it? is the Greek still readable
  when you Read it?) before suspecting LaTeX packages. During this session that symptom was first
  misdiagnosed as "the `float` package is missing and its auto-install blocks" -- wrong on both counts:
  `float.sty` has been installed since 2016 and `\usepackage{float}` + `[H]` builds
  `docs/status_report.tex` in 7.6s for all 3 passes. Don't re-derive that false conclusion.
- **PowerShell gotchas hit in practice**: (1) redirecting a long-running `python` background run with
  `*> file.log` can silently leave the log file at 0 bytes while the process keeps running fine in the
  background -- if a background run's log looks empty after a while, check `Get-Process python` before
  assuming it crashed (it may just be the redirect, not the process); prefer letting the tool capture
  output directly (no shell redirection) for anything backgrounded. (2) an unquoted argument like
  `-output-directory=$scratch` does not always interpolate `$scratch` -- it can pass the literal string
  through and silently create a folder named `$scratch` (verbatim) in the cwd instead of writing to the
  intended temp path. Force interpolation by building the arg as its own quoted variable first (e.g.
  `$outArg = "-output-directory=$scratch"`, then pass `$outArg`) and verify the scratch dir actually has
  files in it before trusting a build "succeeded".
- **Verification ritual for any change to a file many scripts import from, or any script
  consolidation/rename**: back up the affected `results/*.csv` (and `.png`) outputs, rerun, then compare
  with `Get-FileHash -Algorithm SHA256` (byte-identical) or, if that mismatches, `pd.testing.
  assert_frame_equal(..., check_exact=False, rtol=1e-6)` after sorting both frames on a stable key (row
  order can legitimately differ from dict-iteration order or ties in an unstable sort without the
  underlying values being wrong) -- then delete the backups once confirmed. This isn't optional diligence:
  during the `value_betting.py`/`market_devig_comparisons.py` consolidation this exact check caught a real
  behavioral bug (a generalized helper applied an extra round-filter in the one case that never had it
  before, changing `n_bets` counts, not just row order) that would have shipped silently otherwise.

## Architecture

Everything downstream reads from `data/processed/odds_long.csv` (built by `data_processer.py` from raw
football-data.co.uk CSVs in `data/raw/`). Almost every script after that imports shared loading +
algorithm code directly from `sleeping_experts.py` rather than duplicating it -- that file is the one
piece of machinery nearly everything else builds on:

- `load_full_universe()` -- builds the (rounds x experts x outcomes) tensor + "awake" coverage mask,
  sorted chronologically.
- `run_hedge_sleeping()` / `run_ogd_sleeping()` / `run_ftrl_sleeping()` -- the three OCO algorithms,
  generalized to handle bookmakers that don't quote every match (the "Sleeping Experts" reduction:
  per-round renormalization over only the currently-awake subset, frozen state while asleep).
- `evaluate()` -- log-loss / Brier / RPS / accuracy.

`final_ranking.py`'s `calibrate_full_history()` (online temperature scaling, a second OCO stage on top of
the mixture) is itself reused by `significance_test.py`, `value_betting.py`,
`value_betting_online_kelly.py`, `market_devig_comparisons.py`, `contextual_experts.py`,
`pooling_window_comparison.py` and `calibration_per_season.py`. Later "extension" scripts
(`fixed_share.py`, `value_betting.py`, `value_betting_online_kelly.py`, `contextual_experts.py`,
`outcome_class_accuracy.py`, `temporal_trends.py`, `pooling_window_comparison.py`,
`calibration_per_season.py`, `calibration_warmstart.py`, `market_devig_comparisons.py`) are each
self-contained and deliberately NOT merged into `sleeping_experts.py` -- see each file's module docstring
for why. Where an extension needs a slightly different version of a shared function (e.g.
`calibration_warmstart.py` needs a starting temperature), it carries its own small copy rather than
editing the shared file that half the scripts import -- `value_betting.py` and
`market_devig_comparisons.py` each carry their own small `split_by_market_phase()` /
`load_universe_from()` for the same reason, so the two stay independent of each other.

### Non-obvious gotchas worth knowing before extending this

- **Bootstrap/significance sign convention**: `moving_block_bootstrap_test` / `paired_diff` in
  `significance_test.py` define `diff = loss(A) - loss(B)`, so a **negative** mean_diff means the
  first-named series is BETTER (lower loss). Misreading this sign previously produced a wrong
  "calibration hurts on the big dataset" conclusion that had to be corrected across several report
  sections -- double-check the direction before writing any interpretation of a bootstrap-test row.
- **`run_*_sleeping` now skip rounds where NOBODY is awake** (added after `project_to_simplex` raised
  `IndexError: index -1 is out of bounds` from `value_betting_sharp_anchor.py`). This never fires in the
  standard pipeline -- `load_full_universe()` guarantees >=1 quote per round on the full panel, and the
  subset builders filter to >=1 awake -- but it does the moment you drop a column and some match was
  quoted ONLY by that bookmaker. Verified: `results_table.csv` and `sleeping_experts_comparison.png` are
  byte-identical before and after the guard. The affected row of `W` stays all zeros, so a caller that
  evaluates such a round would get a zero mixture probability: restrict betting/evaluation rounds
  yourself, as the value-betting files do.
- **Never renormalize the PERSISTENT weight vector in a Hedge variant** -- only the awake-restricted
  prediction copy may be normalized. `run_hedge_sleeping`'s docstring says this and `fixed_share.py`
  documents an earlier occurrence, and it still got repeated in `learning_rate_study.py`: rescaling `w`
  after each update (there, to stop underflow at large eta) silently produces a DIFFERENT algorithm and
  a flattering set of conclusions. If you need a large eta, carry the weights in **log space** and apply
  a log-sum-exp shift to the prediction only. Whenever you write a variant of one of these algorithms,
  add the assertion that it reproduces the shipped one at its default setting -- that check is what
  caught it.
- **Never read a "final weight" as `W[-1, k]`**: the three `run_*_sleeping` functions write `W[t, i]` only
  for experts AWAKE at round t (asleep coordinates of that row stay 0), so the last row is exactly 0 for
  every bookmaker that didn't quote the final match of the dataset -- 10 of the 24, PSC (the best single
  predictor overall) among them. Read as "final weight" that 0 says "didn't cover the last match", not "the
  algorithm learned to ignore it". Use `sleeping_experts.last_awake_weight(W, awake)` instead (weight at
  each expert's own last awake round; entries are individually interpretable shares but do NOT sum to 1).
  This was a real, shipped bug in three places: it hid the 4 worst frozen-advantage cases from
  `fixed_share.py`'s diagnostic (1XB/1XBC/BF/BFC reported 0.0 at every alpha while actually sitting on
  0.164-0.177) and silently dropped PSC from every per-league "most trusted" list in
  `contextual_experts.py`. Same trap applies to any new per-expert state readout.
- **Sign conventions differ between LOSS tables and GROWTH tables**: the bootstrap gotcha below is about
  log-loss, where lower is better. But `value_betting.py` / `value_betting_online_kelly.py` bootstrap
  log-GROWTH per bet, where HIGHER is better -- so "negative mean_diff = first-named is better" is exactly
  backwards there. Check which quantity a table holds before writing any interpretation of it.
- **football-data.co.uk data quirks already handled** in `data_processer.py`: `VC` (VC Bet) and `BV`
  (BetVictor) are the same bookmaker mid-rebrand and are merged into `VC_BV`/`VC_BVC`; `BbAv`/`BbMx`
  (Betbrain aggregate odds) only exist in pre-2021 seasons and are excluded as non-independent experts
  alongside `Max`/`Avg`; the `Season` column must be read with `dtype=str` (not pandas' inferred int) when
  joining against `load_full_universe()`'s string-typed `seasons` array, or season-keyed lookups silently
  return NaN.
- **Opening vs. closing odds**: a processed bookmaker name ending in `C` is the CLOSING-odds counterpart
  of the identically-prefixed opening one (`B365` opening / `B365C` closing, `VC_BV` / `VC_BVC`) -- this is
  football-data.co.uk's own column convention and `data_processer.py` preserves it. The 24-expert panel
  splits cleanly 12 opening / 12 closing (`market_devig_comparisons.py` and `value_betting.py` both rely
  on this, each carrying its own small `split_by_market_phase()`).
- **Tier A / coverage threshold**: results are only trustworthy for bookmakers with >=70% match coverage
  (`COVERAGE_THRESHOLD` in `final_ranking.py`). Several low-coverage bookmakers are active in exactly one
  season, and their apparent performance is confounded with that season's difficulty, not real skill (see
  the `single_season` flag in `results_table.csv`). Rankings that deliberately include every bookmaker
  (e.g. `opening_vs_closing_ranking.csv`) print a warning listing them.
- **Charts**: the dataviz skill's conventions are in effect for anything plotted -- notably, log-loss
  comparisons use dot/slope plots, not bar charts, because log-loss values cluster near 1.0 with no
  meaningful zero (a truncated-baseline bar would visually exaggerate tiny differences). Plots reuse the
  project's established colors (`COLORS` in `sleeping_experts.py`, `RANK_COLORS` in `final_ranking.py`)
  and shared axis styling (`_style_axes` in `calibration_analysis.py`) instead of inventing new ones.
