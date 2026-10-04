# Work plan / status

Read this first in a new session to pick up where the previous one left off. For "what does file X do
and what functions does it have", see `docs/CODE_OVERVIEW.md` instead -- this file is about *why* and
*what we found*, not *how the code is organized*.

## Thesis

**"Online Convex Optimization based tuning of Mixture of Experts for Sports Outcome Prediction."**

Core idea: instead of training a model on match statistics (which betting sites already do, with far more
data/compute than a thesis has), treat each bookmaker as an "expert" giving H/D/A probabilities, and learn
online (Hedge / OGD / FTRL) a weight vector over experts so the weighted mixture beats any single
bookmaker. This is the classic "prediction with expert advice" / Online Convex Optimization framing.

## Current data

football-data.co.uk, "Main Leagues" section only (the "Extra Leagues" section has a different format and
generally no odds, so it's out of scope). 22 divisions (England x5, Scotland x4, Germany/Italy/Spain/France
x2, Netherlands/Belgium/Portugal/Turkey/Greece x1), 10 seasons (2016/17-2025/26), 76,639 matches of which
**76,584** have at least one usable bookmaker quote (those are the rounds every algorithm runs on),
**26** bookmaker "experts" after cleanup: the VC/BV merge and Betbrain-exclusion (see CLAUDE.md gotchas).

**Interwetten (IW, IWC) was excluded for a while and has been RE-INCLUDED** -- `EXCLUDED_BOOKMAKERS` in
`sleeping_experts.py` is now empty. The exclusion was wrong on two counts. It was not symmetric: the
panel keeps 13 series with 0% coverage in the first eight seasons, the mirror image of IW's pathology,
while IW is the 7th best-covered series of the 26 and was the only one dropped. And it treated the
symptom -- the real defect was a coverage floor computed over the whole decade plus a global
intersection across compared series, and both have since been fixed where they mattered
(`recent_window.py` recomputes its floor inside its own window, `best_configuration.py` intersects only
over the series it compares). Every number in `results/` was regenerated on the 26-expert panel.

Started smaller (5 seasons, 10 divisions) and was expanded mid-project -- several findings changed between
the two scales (see below). `pooling_window_comparison.py` measures this directly.

## What's built, roughly in order

1. **Data pipeline**: `download_data.ps1` -> `data_processer.py` -> `data/processed/{odds_long,odds_long_shin}.csv`
   (produces both de-vig normalizations in one run; supersedes the older `process_data.py` +
   `process_data_shin.py`, now in `scripts/deprecated/`).
2. **Core algorithms**, first prototyped on a 6-bookmaker full-coverage panel (an early draft, not carried
   into this project), then generalized to the full partial-coverage bookmaker panel via the Sleeping
   Experts reduction (`sleeping_experts.py` -- the shared foundation almost everything else imports).
3. **Calibration**: diagnosed a favorite-longshot bias (`calibration_analysis.py`), fixed it with online
   temperature scaling as a second OCO stage (`calibration_correction.py`), then built the single
   authoritative raw-vs-calibrated ranking table computed the same way over the full history
   (`final_ranking.py`).
4. **Statistical rigor**: moving block bootstrap significance testing (`significance_test.py`) -- accounts
   for serial correlation in consecutive match losses, unlike a naive t-test.
5. **Extensions** (each answers one specific follow-up question, each its own file on purpose):
   - `fixed_share.py` -- fixes a "frozen advantage" bug-like behavior in sleeping-experts Hedge (rare
     experts keep inflated weight because nothing erodes it while asleep).
   - `contextual_experts.py` -- does per-league weight specialization help? (No, except Scotland.)
   - `value_betting.py` -- would a real bettor make money? (Leave-one-bookmaker-out Kelly simulation.)
   - `outcome_class_accuracy.py` -- accuracy broken down by Home/Draw/Away specifically.
   - `temporal_trends.py` -- is the market getting more reliable over time? (Tests this directly.)
   - `pooling_window_comparison.py` -- does the 5-year vs 10-year pooling window change conclusions?
   - `calibration_per_season.py` -- does resetting the calibrator every season (instead of one global fit)
     do better, given the market is non-stationary?
   - `calibration_warmstart.py` -- does carrying the previous season's fitted temperature forward (instead
     of resetting to T=1 every season) beat both the global and cold-start-per-season variants?
   - `market_devig_comparisons.py` (`run_opening_vs_closing`) -- does restricting each mixture algorithm to
     only opening-odds bookmakers vs. only closing-odds bookmakers reveal a gap, i.e. does the market's
     final, most-informed price make a better mixture than its first?
   - `value_betting_online_kelly.py` -- does LEARNING the Kelly stake fraction itself online (projected OGD
     on negative-log-wealth, a third OCO stage on top of the pipeline) beat the classical closed-form Kelly
     formula? (No -- see Key findings below.)
   - `value_betting_ogd_vs_ftrl.py` -- does OGD's forecasting-stage edge over FTRL survive into the
     betting simulation? (Weakly and only against closing-side bookmakers -- see Key findings below.)
6. **Methodological robustness checks** (built in a later session, after the extensions above):
   - `data_processer.py` (Shin dataset half) / `market_devig_comparisons.py`
     (`run_shin_vs_basic_comparison`, `run_shin_vs_basic_calibration`) / `value_betting.py`
     (`run_shin_vs_basic`) -- is the whole project's basic-proportional de-vig choice load-bearing?
     Rebuilds the dataset with Shin's (1992) method and re-runs the core comparison, ECE/calibration, and
     value betting on it in parallel.
   - `value_betting.py` (`run_bw_leakage_check`) / `market_devig_comparisons.py`
     (`run_quality_transfer_correlation`) -- found and fixed a look-ahead bias in `value_betting.py`'s
     original BW (opening) test (see Key findings below), plus a side check of whether closing-side
     bookmaker skill predicts opening-side skill (it doesn't, reliably).
   - `value_betting.py` (`run_all_bookmakers_sweep`) -- extends the corrected, non-leaky leave-one-out
     design from a single opening/closing pair (BW, B365C) to every sufficiently-covered bookmaker
     (11 total).
   - `value_betting.py` (`run_closing_trained_both_norms`, `run_training_panel_sweep`) -- generalizes the
     training-panel question to a full 108-row grid (2 normalizations x 3 panel types x every applicable
     target): opening-trained betting against closing targets loses catastrophically; Shin never flips the
     qualitative story anywhere in the sweep.

   (`market_devig_comparisons.py` and `value_betting.py`, above, each consolidate several scripts from an
   earlier session -- `opening_vs_closing.py`, `shin_vs_basic_comparison.py`, `shin_vs_basic_calibration.py`,
   `shin_vs_basic_value_betting.py`, `opening_closing_quality_transfer.py`, `value_betting_all_bookmakers.py`,
   `value_betting_closing_trained.py`, `value_betting_training_panel_sweep.py` -- now in
   `scripts/deprecated/`; see `docs/CODE_OVERVIEW.md` for the exact function-level mapping.)
7. **`docs/thesis.tex`** -- the full Greek LaTeX thesis (see "Where things live" -- this REPLACED the
   older `thesis_report.tex`/`supervisor_report.tex`, now in `docs/superseded/`, per explicit instruction
   to keep only `status_report.tex` from the old write-ups). ~60 pages: title/abstract (EL+EN), related
   work, theory with full proofs, data, methodology, implementation, results (with figures), discussion,
   conclusions, and an auto-generated appendix of every result table. Compile with XeLaTeX (see
   CLAUDE.md). It quotes numbers from `results/` by hand in the body text; the appendix tables are
   regenerated by `scripts/make_appendix_tables.py` (rerun after any pipeline change, then rebuild).
8. **Later additions, each its own standalone experiment**: `recent_window.py` +
   `recent_window_value_betting.py` (every core experiment rerun on the last 5 seasons, where closing
   bookmakers actually clear the coverage floor), `deployment_test.py` (real calendar-date train/test
   split), `markov_ogd.py` (OGD with a Markov transition on the weights, plus an MCGD-in-the-dependent-
   sampling sense variant), `mcmc_kelly.py` (Bayesian/MCMC bet sizing vs. fixed fractional Kelly). All
   negative or null results -- see Key findings.
9. **Three more standalone experiments, added last**: `bandit_oco.py` (what survives when the learner
   sees one scalar per round instead of the whole loss vector), `oco_kelly.py` (learn the Kelly
   MULTIPLIER online rather than the stake -- the one non-rejection of the batch), and
   `regime_switching.py` (a Markov chain over latent market states rather than over the weight vector).
   See Key findings for all three.

## Key findings (current, corrected)

- **APPLIED: all three algorithms now carry a tuned step** (`sleeping_experts.py`):
  `OGD_STEP_MULTIPLIER = 100`, `HEDGE_STEP_MULTIPLIER = 0.25`, `FTRL_STEP_MULTIPLIER = 0.05`.
  The Hedge/FTRL ones are new and came from a methodological error worth remembering: the original
  sweep's grid was 1, 2, 5, ... 1000 and it reported both as "already optimal at 1.0". **1.0 was the
  grid's lower boundary, not an optimum.** Searching downwards found genuine interior minima on the
  train prefix (Hedge 0.998118 at x0.25, FTRL 0.998117 at x0.05, against 0.998128/0.998160 at x1),
  worth +0.00013 and +0.00018 held-out. Effect on the headline: Hedge calibrated 0.99771 -> 0.99768,
  FTRL 0.99776 -> 0.99769. **No qualitative conclusion moved** -- OGD < Hedge < FTRL holds, PSC still
  beats OGD significantly on the full panel, Hedge/FTRL still lose to the uniform average 8 of 8 in the
  restricted panels.
- **REVERSED: Interwetten is back in the panel** (`EXCLUDED_BOOKMAKERS` is now empty), 24 -> 26
  experts, 76,583 -> 76,584 rounds. See "Current data" above for why the exclusion was the wrong fix.
  Effects of re-inclusion, all small but several qualitative:
  - OGD raw 0.99739 -> 0.99745, calibrated 0.99695 -> 0.99693.
  - IW enters Tier A as the **worst-ranked** bookmaker (raw 1.00186) despite the 7th best coverage,
    which makes "we beat everyone but PSC" slightly cheaper -- say so when quoting it.
  - Value betting gains a **second significant profit** (IW, 7.89x, p=0.008), and PSC turns from a tie
    into a **significant loss** (0.0002x, p<0.001) -- the expected direction, since PSC is the best
    forecaster in the panel and an inferior model should lose to it.
  - In the deployment test, **BFEC now beats the mixture significantly** on the full-panel variants
    (+0.00089, p=0.013 online; +0.00143, p=0.002 frozen). "We tie every closing price" now holds only
    for the closing-only variant.
  - In the no-Pinnacle closing-only panel OGD moves from #5 to **#1**, though the uniform average is
    0.00002 behind it, i.e. still practically indistinguishable.
  - **The tuned step multipliers moved.** Re-running `learning_rate_study.py` on 26 experts puts the
    held-out minimum at x0.05 for Hedge and x0.005 for FTRL, against the shipped x0.25 and x0.05; OGD's
    x100 is still right. They were deliberately NOT re-tuned so all `results/` numbers stay mutually
    comparable, so Hedge and FTRL are reported slightly conservatively. Re-tuning = another full run
    plus another pass over the write-ups.
- **Four further optimisation ideas were tried and all REJECTED.** Recording them so they are not
  re-derived: (1) Fixed-Share alpha swept jointly with the Hedge step -- alpha>0 loses at every step,
  because alpha and the step both control "how fast it forgets" and alpha only ever compensated for a
  step that was too large. (2) Step and calibration rate swept jointly instead of in two stages -- no
  interaction whatsoever (best calibration rate is 0.15 in 11 of 11 rows), and joint train selection
  picks x150 for OGD, which is worse held-out. (3) A per-expert step clock, using each expert's own
  awake-round count instead of global t, which is what the Sleeping Experts regret bound is literally
  about -- **better on train, -0.00073 held-out**, because a 6.9%-coverage expert gets a ~14x larger
  step and the mixture chases single-season bookmakers. The cleanest overfitting demonstration in the
  project. (4) AdaGrad, per-expert and scalar -- recovers 20% at best.
  **Still open and genuinely untried**: a constant, non-decaying step. The old sweep only tried
  constants >= 0.1, which the new results show is far too large.

- **APPLIED: OGD's step multiplier is now 100** (`sleeping_experts.OGD_STEP_MULTIPLIER`, applied after
  the study below; all results in `results/` are post-change). Effects on the headline numbers, all
  verified by rerunning the pipeline: OGD raw 0.99826 -> **0.99745**, calibrated 0.99760 -> **0.99693**;
  Hedge and FTRL unchanged (their own sweeps put them at 1.0, so the three-way comparison stays fair).
  **PSC still significantly beats OGD but by 2.6x less** (calibrated paired diff +0.00114 -> +0.00043,
  raw +0.00149 -> +0.00065, both still p<0.001 on the 71,818 common rounds -- note the ranking table's
  apparent near-tie is an artifact of OGD being scored on all 76,584 rounds). NEW: **OGD now
  significantly beats B365C** (-0.00063, p<0.001; previously -0.00022, p=0.091, not significant), and
  its margins over Hedge/FTRL/Uniform roughly quadrupled (-0.00100 / -0.00104 / -0.00141, all p<0.001).
  Per-league specialisation still loses, by more than before (+0.00058, p<0.001).
- **The closing-odds advantage is a top-quintile phenomenon, not a general property of closing prices**
  (`information_arrival.py`). Stratifying by how much the line actually MOVED between open and kickoff
  (total-variation distance, averaged over the 13 opening/closing pairs), the closing-minus-opening gap
  runs -0.00044 (Q1, **p=0.052, not significant**), -0.00072, -0.00200, -0.00452, **-0.01158** (Q5) --
  a **26x** gradient. In the fifth of matches where the price barely moved there is no measurable
  closing advantage at all. Shin's `shin_z` (already computed per match by `data_processer.py`, never
  analysed until now) is a far weaker stratifier: -0.00293 to -0.00410, 1.4x.
  QUALIFICATION to state in the write-up: movement is measured from the same closing prices the closing
  panel uses, so Q1's null is partly mechanical -- if the prices barely differ the mixtures barely can.
  The non-trivial half is the direction at the top: a large move could go either way, and the later
  price is systematically the better one. Control: raw difficulty is NOT monotonic across the strata
  while the advantage is, so this is not a "harder matches" artifact.
  Framing for the thesis: market microstructure (informed vs uninformed traders, spread as protection --
  Kyle 1985, Glosten & Milgrom 1985, Shin 1992), NOT "insider detection" -- odds alone cannot separate
  insider from public information, only show that information arrived.
- **ADOPTED: the closing-only panel is the project's best configuration, and it makes the mixture
  statistically TIED with PSC** (`best_configuration.py`). Four wins had been measured separately and
  never stacked: the corrected OGD step, closing-only beating the full panel, Shin beating proportional
  de-vig on raw log-loss, and calibration. Every configuration is scored on the SAME 71,818 rounds (the
  intersection of all configs' coverage with PSC's) with a held-out chronological 25% tail, because
  picking a config by reading these numbers IS selection. Calibrated: PSC 0.99685, **A (closing+Shin)
  0.99692**, closing+basic 0.99693, baseline (full 26, basic) 0.99728, full+Shin 0.99731.
  - **The panel does all the work**: A vs baseline **-0.00036, p<0.001**; closing+basic vs baseline
    -0.00034, p<0.001.
  - **Shin alone adds nothing measurable**: full+Shin vs baseline +0.00004, **p=0.181** -- consistent
    with Shin and temperature scaling correcting the same favorite-longshot bias.
  - **vs PSC**: baseline loses significantly (+0.00043, p<0.001); **A ties (+0.00007, p=0.234)**,
    closing+basic ties (+0.00009, p=0.143). This is the substantive change to the project's headline.
  - Config B (A minus single-season bookmakers) is marginally worse on the held-out tail; rejected.
  LIMITS TO STATE WHENEVER QUOTING THIS: the point estimate still favours PSC -- "tied" means the
  difference stopped being significant, not that the mixture won. On the held-out tail all rows sit
  within 0.0001 of each other, so the tail confirms the tie rather than picking a winner. And every
  closing configuration has PSC INSIDE the mixture, so it reads "the blend reaches its own best member",
  not "we reached it from outside".
- **Without Pinnacle: the full panel result holds, the closing-only one collapses** (`no_pinnacle_panel.py`,
  standalone -- the main pipeline keeps all 26 experts). Rerun with the same discipline as
  `best_configuration.py` (one common round set for every series + a 75/25 chronological split), which
  matters: the earlier loose version scored each series on its own coverage and reported a win over
  B365C on the closing panel that did NOT survive.
  - **Full panel** (24 experts, 100% round coverage, 54,877 common rounds): all four algorithms rank
    above all five surviving Tier-A bookmakers -- Hedge 0.99772, FTRL 0.99778, OGD 0.99782, uniform
    0.99837, vs VC_BV 0.99987 -- and **OGD significantly beats 5 of 5** (-0.00181 to -0.00315, all
    p<0.001). OGD is also first on the held-out tail. This answers "isn't the mixture just reproducing
    Pinnacle?": the one-bookmaker caveat is about one exceptional odds-setter, not the method.
  - **Closing-only panel** (12 experts, 32,210 common rounds): OGD ranks #1 (0.99919) but beats only
    **3 of 5** significantly (WHC -0.00088, IWC -0.00077, BWC -0.00051); B365C p=0.505 and VC_BVC
    p=0.284 are ties, and VC_BVC is ahead on the held-out tail. The telling number: OGD (0.99919) is
    essentially IDENTICAL to the uniform average (0.99921), with Hedge (0.99923) and FTRL (0.99932)
    behind it. Without Pinnacle the closing bookmakers are too similar for the weighting to have any
    quality difference to learn -- the same mechanism as every homogeneous panel here. Say it plainly:
    the best configuration is best partly BECAUSE it contains the market's best odds-setter.
  DO NOT compare a with-Pinnacle number against a without-Pinnacle one directly -- the round sets differ
  (76,584 vs 54,877 vs 32,210). An earlier draft claimed "Pinnacle's contribution is 0.99693 -> 0.99793"
  built exactly that way; it was not measuring what it claimed and has been removed from all reports.
  `no_pinnacle_panel.py` now ALSO runs a third, opening-only panel (12 experts, symmetric with the
  closing-only one, since PS/PSC are the same company at two points in time). Same mechanism as closing:
  the uniform average (0.99973) beats OGD (0.99988) there too. More importantly: **OGD loses
  significantly to VC_BV** on that panel (+0.00025, p=0.009) -- the ONLY time anywhere in the project
  that a non-Pinnacle bookmaker beats the mixture significantly. State this plainly wherever the "we
  only lose to Pinnacle" claim is made.
- **`recent_window.py`: every core experiment rerun on the last 5 seasons (2021/22-2025/26, 38,749
  rounds)**, because the 10-season coverage floor lets through only ONE closing bookmaker (PSC) while 5
  closing bookmakers (B365C, BWC, VC_BVC, WHC, IWC -- wait IW is excluded, so 4: B365C/BWC/VC_BVC/WHC)
  clear it in the shorter window, making the hard closing-vs-closing comparison possible at all.
  OGD ranks #2 of 14 in the full panel (0.99488 vs PSC 0.99439), ahead of every closing bookmaker except
  Pinnacle; the PSC gap shrinks to +0.00029 (p=0.032) from +0.00044 in the 10yr panel. Closing-only still
  beats the full panel significantly. **Two findings REVERSE here and must be stated**: Hedge and FTRL
  lose to the uniform average in ALL THREE panels in this window (not just the restricted ones, as in the
  10yr result), and log-loss and Brier stop agreeing on the ranking order. A live-market section
  (`section_live_market`) also recomputes the coverage floor INSIDE the chronological tail alone, distinct
  from `deployment_test.py`'s real calendar-cut version -- don't conflate the two, they answer related but
  different questions (one is "5-year pooled window", the other is "trained-to-date, scored going
  forward").
- **`markov_ogd.py`: giving OGD a Markov transition on the weight vector (uniform/pair/phase leak
  kernels) does not help -- every kernel selects alpha=0 on train**, i.e. collapses to plain OGD. Verified
  by an assertion that the identity kernel reproduces `run_ogd_sleeping` exactly. Consistent with the
  Fixed-Share result above: once the step itself is tuned, no way of moving weight mass between experts
  post-hoc adds anything, because they all attack the same underlying quantity. A SEPARATE thing tried in
  the same file, following arxiv.org/abs/1809.04216 (Sun, Sun & Yin 2018) -- MCGD in the sense of
  DEPENDENT SAMPLING rather than a weight-transition kernel: if the moving-block-bootstrap dependency
  horizon is tau ~ T^(1/3) ~ 42, the step should be inflated by ~sqrt(tau) on top of the usual bound-
  slack correction. The two corrections combine to an inflation factor of ~24 from theory, against the
  empirically-tuned x100 -- a partial but independent theoretical explanation for why the worst-case step
  was too conservative, arrived at without any grid search.
- **`mcmc_kelly.py`: posterior-expected (Bayesian/MCMC) Kelly sizing does not beat fixed fractional Kelly,
  and sometimes loses to it significantly.** The idea: treat the outcome distribution theta as uncertain
  (a Dirichlet posterior conditioned on the awake experts' forecasts and their learned weights) and
  maximise EXPECTED log growth under it, so stake shrinkage emerges from disagreement instead of being an
  arbitrary constant. Paired against fixed 1/4-Kelly on the SAME rounds, for 4 target bookmakers (WHC,
  PSC, BWC, B365C): 6 of 8 comparisons are ties, 2 are SIGNIFICANT LOSSES for MCMC (PSC at both full and
  1/4 Kelly, p<0.001; BWC at full Kelly, p=0.028), zero significant wins anywhere. The selected kappa
  (how tightly experts are assumed to cluster around the truth) is also unstable across bookmakers (200,
  200, 20, 20) with no obvious pattern, a sign the selection is fitting noise rather than real structure.
  Filed alongside Fixed-Share and the per-expert step clock as a theoretically-more-principled idea that
  the arbitrary baseline (1/4 fixed) simply beats in practice.
- **OGD's step size is ~100x too small -- the single biggest free improvement found so far**
  (`learning_rate_study.py`). The shipped schedules use `M = log(1/EPS) ~ 13.82` as the bound on a
  round's per-expert loss, but real de-vigged probabilities bottom out near 1-2%, so `eta ~ 1/M` is
  3-4x smaller than an honest bound allows. Measured on a held-out tail: **Hedge is already optimal**
  (multiplier 1.0 wins, every increase is worse), but **OGD improves monotonically from 0.99803 at x1
  to 0.99765 at x100**, then degrades (x200 0.99778, x1000 0.99895) -- a genuine interior optimum that
  train and held-out agree on. **+0.00038 held-out, comparable to the entire OGD-vs-uniform advantage
  (0.00061) and to the whole calibration stage (~0.00065)**, for the cost of one constant. APPLIED --
  see the first bullet of this section for the post-change headline numbers.
  Counter-intuitive companion result: the theoretically SUPERIOR Bayesian / constant-eta=1 variant
  (log-loss is exp-concave, so the Aggregating Algorithm gets regret <= ln N instead of sqrt(T ln N))
  performs BADLY -- held-out 1.0010, worse than uniform averaging. The regret bound is against a STATIC
  comparator and this panel is not static: PSC's coverage collapses from ~100% to 38.8% in the final
  season, which sits inside the held-out tail, and a fast-concentrating posterior is exactly what a
  vanishing expert punishes. Good concrete evidence for the non-stationarity argument that motivates
  `fixed_share.py`.
- **OGD is the best of the three core algorithms**, and the only one that *originally* beat naive uniform
  averaging significantly -- but with the full 10-year dataset, Hedge and FTRL beat uniform averaging
  significantly too (this reversed when the dataset grew; see `pooling_window_comparison.py` for the
  isolated time-depth effect).
- **PSC (Pinnacle closing) has a statistically significant edge over OGD in the shipped full-panel
  configuration**, before and after calibration, clearer on the full 10-year dataset than it looked on
  the original smaller one. In the ADOPTED closing-only configuration that edge is no longer significant
  (p=0.234) -- see the `best_configuration.py` bullet above. Both statements are true of different
  panels; never quote one without saying which panel it is about.
- **Online temperature scaling (calibration) helps significantly**, consistently, on both the original and
  the expanded dataset. *(Correction: an earlier session mis-read the bootstrap sign convention and
  concluded the opposite -- that calibration started hurting on the big dataset. That was wrong; see the
  CLAUDE.md gotcha.)* Current numbers (`results/significance_test_table.csv`): raw minus calibrated log-loss
  is +0.00065 to +0.00073 for all four algorithms, every one significant.
- **Per-season calibration** (resetting the temperature fit every season) is significantly better than raw,
  but statistically indistinguishable from the existing single global calibration -- the smaller per-season
  sample (~7,658 rounds vs 76,584) likely cancels out the benefit of tracking drift.
- **Warm-start per-season calibration** (carry the previous season's fitted T forward instead of resetting
  to T=1, but still reset the round counter so eta_t stays responsive) visibly removes the cold-start
  "sawtooth" in the per-season T trajectory (`results/calibration_warmstart.png`), and its log-loss lands
  between the global and cold-start-per-season variants for every series -- but it is statistically
  indistinguishable from BOTH (all `warm-start vs global` / `warm-start vs per-season` comparisons
  non-significant in `results/calibration_warmstart_significance.csv`), while still beating raw
  significantly almost everywhere. So it's a genuine compromise point, not a strict improvement over
  either existing variant -- removing the cold-start cost didn't translate into a measurable log-loss win
  at this dataset size.
- **Fixed-Share** fixes the frozen-advantage problem (CLC: 0.240 -> 0.064 and CL: 0.222 -> 0.063 at
  alpha=0.01; the four 2024/25-only bookmakers 1XB/1XBC/BF/BFC from 0.164-0.177 down to exactly 0.100,
  i.e. parity within their last round's 10-strong awake set) at a small but real cost to overall log-loss
  (0.99845 -> 0.99885, i.e. slightly WORSE, not better; `results/fixed_share_table.csv`).
- **Per-league specialization doesn't help** (pooled beats per-league specialists significantly overall:
  mean diff +0.00049, 95% CI [+0.00005, +0.00091], p=0.031), except Scotland where the specialist wins by a
  hair (0.93663 vs 0.93674). Scoped to the 6 top-flight leagues (E0, SC0, I1, SP1, D1, F1; 2,231-3,800
  matches each), not all 22 divisions. Plausibly because per-league sample size is too small for the online
  algorithm to converge well -- same bias-variance story as per-season calibration, in a different dimension
  -- and because the specialists end up trusting roughly the same experts the pooled model does (PSC #1 in
  4 of 6 leagues, B365 in the other 2), rather than discovering anything league-specific.
- **Draws are essentially unpredictable as an argmax pick** for every algorithm/bookmaker (<0.7% accuracy),
  even though draws are ~26% of matches -- a well-known football-forecasting phenomenon, confirmed here.
  **But that is a statement about RECALL, not about predictability** (`outcome_class_comparison.py`): on
  the rare occasions a series does call a draw, its PRECISION is 35-56% against a 26.2% base rate, for
  every single series -- even the weakest estimate (PSC, 35.5% on n=369 picks) is 3.7 standard errors
  above the base rate. A draw is almost never anyone's single most likely outcome; that is not the same
  as the call being uninformative when it is made. Worth phrasing carefully in the write-up.
- **Per-class accuracy IS meaningful -- as evidence the calls beat chance, NOT as a way to rank methods**
  (`outcome_class_accuracy_test.py`, the test the two descriptive outcome-class files were missing).
  **32 of 33 series x class cells beat their own base rate at 95%**: Home 51% vs a 43.5% base rate, Away
  **48.6% vs 30.3%** (a 1.6x lift -- the most informative of the three calls), Draw 35-56% vs 26.2%. The
  one failure is Uniform average on draws (CI low 25.8% vs a 26.2% base rate), which also calls draws
  least often of the four algorithms. So "accuracy is useless" -- true pooled, 0 of 18 -- is too strong:
  conditioned on a call being MADE, it is very far from chance. But it still cannot rank the methods:
  23 of 33 recall comparisons look significant, yet recall differences are driven by how often each
  series picks that class (OGD calls draws 161 times to B365's 59, so it "wins" draw recall and "loses"
  home recall), the same tilt artifact as per-class log-loss; and on precision, the fair comparison, the
  CIs overlap almost entirely for Home and Away and are far too wide for Draw (n=45-369).
- **OGD's edge over Hedge/FTRL is a home/away split, not uniform skill** (`outcome_class_comparison.py`):
  significantly better on home wins (-0.00112 / -0.00118, p<0.001), significantly WORSE on away wins
  (+0.00119 / +0.00125, p<0.001), tied on draws; the pooled win comes from home wins simply being the
  commonest outcome (33,280 vs 23,224 rounds). Mechanism measured directly: OGD assigns +0.0003 more
  probability to home and -0.0004 less to away than Hedge/FTRL on average. That lean is TOWARD the true
  base rates (home 43.46% actual, all four algorithms predict ~43.0%, OGD least biased), so it is a
  correction rather than a defect -- but it does mean "OGD is the best algorithm" is a statement about
  the outcome mix, not about uniform superiority.
  Related and useful for the write-up: **PSC's advantage over the mixture is entirely in home wins**
  (+0.00521 there, tie on draws) and **OGD significantly beats PSC on away wins** (-0.00271). Across
  bookmakers, away wins are where the mixture is most reliably ahead (5 of 7, no losses), draws where
  the margins are largest but mixed, home wins where it is weakest (3 wins, 3 losses).
  METHOD WARNING baked into that file: per-class log-loss rewards a lean toward the class (restricted to
  home wins the loss is just -log(p_H)), so each per-class test is paired with its pooled counterpart and
  flagged `consistent_with_pooled`; 7 of the 25 significant per-class results point the opposite way to
  pooled and are therefore tilt, not skill.
- **No significant trend in market accuracy (log-loss/ECE) over 10 seasons**, but overround (bookmaker
  margin) *increases* significantly (~4.4% -> ~6.0%) -- the opposite of a "maturing, more competitive
  market" story.
- **Value betting -- CORRECTED finding** (the original "large edge against BW" was a look-ahead-bias
  artifact, found and fixed in a later session): `value_betting.py`'s original leave-one-bookmaker-out test
  against BW (an opening-side bookmaker) excluded only BW's own column, so the refitting panel still
  included every closing-side bookmaker -- including BW's own closing counterpart (BWC) -- for the same
  match. In reality none of that closing-side information exists yet when only an opening price is posted
  (all bookmakers' closing prices settle near kickoff, regardless of when each one opened), so this
  overstated the edge by assuming access to future information. `value_betting.py`'s `run_bw_leakage_check`
  fixed it by restricting the panel to opening-only bookmakers: the "huge" edge against BW (final bankroll
  ~10^6-10^9x, p<0.001) disappeared entirely (honest raw: -0.00010, p=0.169; honest calibrated: +0.00004,
  p=0.439). `run_all_bookmakers_sweep` then extended the corrected design to 11 bookmakers
  (coverage >=50%, multi-season): 3 of 11 show a significant *calibrated* edge -- **positive** (profitable)
  against WHC (+0.00056, p<0.001) and IW (+0.00028, p=0.008); **negative** (a real, significant loss)
  against PSC (-0.00021, p<0.001), with PS now just short of significance (-0.00005, p=0.055) -- PSC's negative result is an expected sanity
  check, since it's already the #1 predictor overall, so betting against it with an inferior model should
  lose. The other 7 (B365, BW, WH, BWC, VC_BV, VC_BVC, B365C) show no significant edge either way. So the
  honest framing isn't "no edge vs. sharp, big edge vs. weak" -- it's "no single sharp/weak story; edge
  existence and sign depend on the specific bookmaker." Account limits/liquidity still aren't modeled
  either way. The full training-panel grid (`run_training_panel_sweep`, 108 rows) confirms opening-trained
  betting against CLOSING targets loses catastrophically and consistently (bankroll -> ~0, p<0.0001, all 5
  closing targets x both normalizations) -- stale opening-only information is a bad training panel against
  a much sharper closing price, valid but not profitable.
- **Learning the Kelly stake itself online does not beat the classical formula** (`value_betting_online_kelly.py`):
  framed as a third OCO stage on top of the pipeline (mixture weights -> calibration -> stake size, the
  "Online Portfolio Selection" idea, Cover 1991), a projected-OGD stake learner loses to the classical
  closed-form Kelly formula on the identical bets in 17 of 20 comparisons (significantly worse; the rest
  still worse, just not significantly). The online learner starts blind (f=0) and must learn purely from
  realized win/loss outcomes at a conservative, worst-case-safe learning rate, while the closed-form
  formula uses the already-available probability estimate directly with no cold-start cost -- suggesting
  OCO's value here is in the forecasting stage, not in re-learning what the closed-form answer already
  gives for free.
- **OGD's forecasting edge over FTRL barely survives into the betting simulation**
  (`value_betting_ogd_vs_ftrl.py`, the two algorithms on the identical 11-bookmaker honest leave-one-out
  design, paired per active round): OGD is ahead in 15 of 22 comparisons, but only 4 are significant --
  and all 4 favour OGD, none FTRL (B365C raw p=0.045, BWC raw p=0.020, WHC raw p=0.003, WHC calibrated
  p=0.015). The whole effect sits on CLOSING-side targets (OGD ahead in 9 of those 10 comparisons); on
  opening-side targets the two split 6-6 with nothing significant. Magnitudes: the paired differences are
  ~2-3e-05 log-growth per round, an order of magnitude below the edges themselves, so the choice of
  TARGET matters far more than the choice between these two algorithms -- and FTRL is absolutely better
  on some targets (IW calibrated +0.00034 vs OGD +0.00030, both significant). The script asserts its own
  OGD half reproduces `value_betting_all_bookmakers_table.csv` exactly, so the algorithm is the only
  difference between the two halves.
- **De-vig method choice (Shin vs. basic proportional normalization) doesn't change the story**: rebuilt
  the whole dataset with Shin's (1992) method (models an insider-trading proportion, shifts mass from
  longshot to favorite instead of splitting the bookmaker's margin evenly) and re-ran the core comparison.
  Ranking is IDENTICAL either way (PSC #1, then OGD>Hedge>FTRL>Uniform, then the rest). Shin significantly
  improves RAW log-loss and Brier everywhere (p<0.0001, 11/11 entities) and roughly halves ECE (-35% to
  -57%), but the advantage mostly evaporates after calibration (log-loss stays significant for only 4/11,
  Brier for 0/11) -- because Shin and temperature-scaling calibration correct the same favorite-longshot
  bias via different mechanisms (per-match cross-sectional vs. online across-history), so doing one makes
  the other largely redundant (calibration itself stops being significant for 9/11 series once Shin is
  already applied, vs. 11/11 significant under basic normalization). Same pattern confirmed in the value-
  betting simulation for the two weakest algorithms (FTRL, Uniform average).
- **Closing odds significantly beat opening odds** for every one of the 4 mixture algorithms
  (`results/opening_vs_closing_significance.csv`, all 4 comparisons significant, closing lower log-loss by
  ~0.0034-0.0036) -- consistent with the standard efficient-markets expectation that the closing line has
  absorbed more information (line movement from money placed between market open and kickoff) than the
  opening line. The panel splits cleanly into 13 opening / 13 closing bookmakers by football-data.co.uk's
  own naming convention (a "C" suffix marks the closing counterpart of the identically-prefixed opening
  bookmaker), with no unpaired names.
- **The learned mixture only beats naive uniform averaging when the expert panel is HETEROGENEOUS**
  (`results/opening_vs_closing_vs_uniform.csv`, `_shin.csv`; 3 algorithms x 3 panels x 2
  normalizations = 18 bootstrap tests). On the FULL 26-bookmaker panel all three algorithms beat
  uniform significantly (6/6 tests, as `significance_test.py` already found). But restricted to ONE
  market phase -- where the experts are far more homogeneous in quality -- **Hedge and FTRL lose
  SIGNIFICANTLY to plain uniform averaging in 8 of 8 tests**, and OGD only manages a statistical tie
  (4/4 non-significant, p=0.36-0.55). Both normalizations give the same picture. Reading: the full
  panel's advantage comes from there being a real quality spread to learn (13 closing bookmakers vs 13
  systematically worse opening ones); remove the spread and the adaptation cost (weights moving on
  noise) is no longer paid for. This reframes the project's headline "OGD beats uniform" result --
  it is a statement about the PANEL, not about the algorithm in isolation -- and it is the sharpest
  argument in the project for why OGD is preferable to Hedge/FTRL (it is the only one that doesn't
  actively hurt on a homogeneous panel).
  The algorithm ORDER itself (OGD < Hedge < FTRL) is identical in all 6 panel x normalization cells;
  what moves is the uniform baseline, from 4th on the full panel to 2nd on closing-only and 1st on
  opening-only.
- **Anchoring on the best single bookmaker beats the mixture on the CLOSING side but NOT on the opening
  side** (`value_betting_sharp_anchor.py`, both phases). Closing anchor PSC: ahead on **5 of 6** targets
  (WHC 88.7x vs 39.3x, IWC 8.2x vs 6.9x, BWC 2.7x vs 1.8x; it loses only on BFEC, 1.20x vs 1.67x).
  Opening anchor VC_BV: ahead on only 4 of 7 -- a coin flip. Across all 52 paired tests only 9 are
  significant (4 anchor-calibrated, 3 anchor-raw, 2 OGD-calibrated), so this remains suggestive rather
  than demonstrated. The mechanism is the anchor's own margin in its phase: PSC beats
  OGD (closing) by 0.00031 in log-loss, VC_BV beats OGD (opening) by 0.00008, four times less.
  Anchoring pays exactly to the extent one expert dominates its panel -- the same lesson as the
  uniform-baseline result and the per-panel ranking. Also: the opening side is barely profitable for
  anyone (of 7 targets only IW pays, 14.0x anchor / 8.2x mixture, both p<0.01; PS loses significantly
  for both; the rest sit near 1.0), whereas the closing side has WHC at 88.7x and IWC at 8.2x. So the
  economic case for this whole project lives on the closing panel.
- **The sharp bookmaker used as the forecast beats the learned mixture economically -- on every target,
  but never significantly** (`value_betting_sharp_anchor.py`). This is the standard professional recipe
  and the natural challenge to the thesis: skip the learning, treat PSC's de-vigged closing price as
  truth, bet it against softer CLOSING books (opening targets are inadmissible -- PSC is a closing
  price, so that would be the look-ahead bias again). Result: PSC calibrated finishes ahead of the
  calibrated mixture on **6 of 6 targets**, sometimes hugely (WHC 88.7x vs 34.2x, IWC 8.2x vs 3.8x),
  yet **no individual paired difference is significant** (best p=0.052). A sign test across targets
  gives p=0.031, but they share matches and the same PSC forecasts, so treat that as suggestive only.
  Standalone, PSC turns a significant profit against IWC (+0.00034, p=0.002) and WHC (+0.00057,
  p<0.001). IMPORTANT framing: the mixture already CONTAINS PSC (~0.14 weight on a closing target), so
  this is "PSC alone vs a blend that dilutes PSC with 24 lesser experts", not "sharp vs learned" -- the
  same lesson as the uniform-baseline finding, that blending only pays when no single expert dominates.
  This belongs in the write-up as the strongest honest counter-argument to the economic case for the
  mixture, alongside the caveat that WHC and IWC are the softest books and the first accounts to be
  limited.
- **Skipping the leave-one-out (circularity) DESTROYS the value-betting experiment rather than inflating
  it** (`value_betting_circular.py`, the same A/B as `run_bw_leakage_check` but for all 11 targets, only
  `exclude_target` flipped). This is the opposite of the project's look-ahead bug, and the reason is
  mechanical: the target's own de-vigged price goes into the mixture, pulling `p_hat` toward the very
  price being bet against, while a bet still needs `p_hat * odds > 1` against that bookmaker's VIGGED
  odds -- in the limit the two cancel and nothing ever clears. Measured: **bets fell in 11 of 11
  targets, median -35%**, and the fall tracks the mixture's weight on the target almost perfectly
  (Pearson r = -0.83). The headline number is that weight: on an opening-only panel OGD concentrates
  hard and gives the target **0.72-0.89** (B365 0.887), so "betting against B365" becomes "betting
  against B365 with a forecast that is 89% B365" -- bets -68%, bankroll 0.36 -> 0.95, i.e. flat.
  Closing targets sit on the full 25-expert panel, get 0.03-0.29 weight, and barely move (bets -3%).
  BW and WH flip sign outright; significant results drop 4 -> 3. Useful framing for the write-up: the
  leave-one-out is not there to stop the numbers looking too good, it is there because without it you
  are testing a price against a blend of itself. (Incidental but worth knowing: OGD putting ~0.89 of
  its weight on a single expert is itself a finding about how concentrated the learned mixture gets on
  a small panel.)
- **Betting only inside the "good" probability band does not help** (`value_betting_by_band.py`, the
  obvious follow-up to the band analysis below). With the band chosen HONESTLY -- on the first 75% of
  each target's bets, then scored only on the rest -- it beats the unrestricted strategy on 6 of 11
  targets, a coin flip; the single significant holdout result is BWC at p=0.025 in the LOSING
  direction. The full-sample pass looks better (10 of 99 band-cells significant) but that is roughly
  the ~5 expected by chance from 99 tests at 95%, with mixed signs. The decisive evidence is that the
  train-chosen bands are scattered across the entire spectrum (0-10% through 80-90%, only 30-40%
  repeating, 3 times): a genuinely better region would be selected consistently. What band filtering
  actually does is cut bet counts ~6x and compress outcomes toward 1 (PS 0.39 -> 0.96, WH 1.70 -> 1.11)
  -- variance reduction, not edge. Worth keeping in the write-up as a deliberate negative result: it
  closes off the obvious "why not just bet where you forecast best?" question, and it demonstrates the
  train/val discipline on a question where post-hoc selection would otherwise have produced a
  confident, wrong answer. Underlying reason: a log-loss edge and a betting edge are different things
  -- being better calibrated than a bookmaker does not mean it offers a price worth taking.
- **Stratifying by probability band does NOT rescue accuracy, but it does localise where the mixture's
  edge lives** (`accuracy_by_band.py`). Three views, and they disagree in an informative way:
  (a) on CALIBRATION a bookmaker wins in all 10 probability bands -- the mixtures are systematically
  worse calibrated than individual bookmakers everywhere, since linear pooling shrinks probabilities
  toward the middle (precisely what the temperature-scaling stage later undoes, which is a cleaner
  justification for that stage than the pooled ECE number);
  (b) on ACCURACY the series stay indistinguishable band by band, and IW is the trap -- best accuracy in
  almost every band, worst Tier-A log-loss overall (and the per-band accuracies are not on common
  rounds, so coverage confounds them);
  (c) on the properly PAIRED per-band log-loss test, OGD beats every bookmaker except PSC in
  essentially every band (31 of 42 cells significant for OGD, 0 against it except PSC, which wins all
  6 bands; the 5 ties are all in the thinnest high-confidence bands). Both margins GROW with
  confidence -- OGD vs IW from -0.0027 (33-40%) to -0.0079 (80-100%), PSC vs OGD from +0.0012 to
  +0.0065 -- so the confident predictions decide everything, while the 33-50% bands (~60% of matches)
  are where every method is nearly identical. Useful for the write-up: PSC's edge over the mixture is
  not uniform, it is concentrated exactly where the mixture is most confident.
- **Brier reproduces every log-loss verdict; accuracy reproduces none of them** (the same vs-uniform
  battery rerun on all 3 metrics -- 54 tests). Brier agrees with log-loss on **18 of 18** verdicts
  (same winner, same sign, same significance, both normalizations) -- expected, both are proper scoring
  rules over the whole distribution. Accuracy produces **0 significant results out of 18**
  (p = 0.098-0.985), with CIs of roughly +-0.0003 to +-0.0013, an order of magnitude wider relative to
  the differences being measured. So the alternative orderings accuracy shows (e.g. FTRL/Hedge ahead of
  OGD on the full panel, OGD LAST on opening-only) are noise, not a competing finding -- worth stating
  explicitly in the write-up, because those orderings look like a contradiction if quoted without their
  p-values. This is the project's empirical justification for using log-loss as the primary metric
  rather than a theoretical preference: at ~50.6% accuracy for every method, a metric that discards
  confidence discards exactly the signal needed to separate them.
- **The opening/closing/full split holds under BOTH de-vig methods**: the same 4 algorithms x 3 panels
  comparison rerun on the Shin dataset reproduces the ordering exactly (closing < full < opening, all 12
  comparisons significant there as well, i.e. 24 of 24 across both normalizations). Shin is uniformly
  ~0.0005 lower in every one of the 12 cells -- a level shift, not a reordering. So neither "closing
  wins" nor "opening odds are a net drag" depends on the normalization choice
  (`results/opening_vs_closing_table_shin.csv`, `_significance_shin.csv`).
- **Closing-only mixtures beat the full mixed 26-bookmaker panel too** (not just opening-only) -- and
  opening-only significantly UNDERPERFORMS the full panel -- for all 4 algorithms (bootstrap-confirmed,
  `results/opening_vs_closing_significance.csv`'s "closing vs full" / "opening vs full" rows). So the
  ordering closing < full < opening (lower log-loss = better) holds with statistical confidence, meaning
  the opening-odds bookmakers are a net drag on the mixture rather than useful diversification. Combined
  into one ranking with every individual bookmaker (`results/opening_vs_closing_ranking.csv`), OGD
  (closing) is the best-ranked algorithm variant at #3 overall, behind only PSC and one single-season
  closing series -- ahead of every other bookmaker and every opening/full algorithm variant.
- Several **data-quality issues found and fixed** by auditing against football-data.co.uk's own
  documentation: VC->BetVictor rebrand merge, Betbrain aggregate-column exclusion (only visible once the
  dataset went back past ~2021), single-season bookmakers flagged as unreliable comparisons.
- **A weight-readout bug found and fixed** (a code review pass, not a data issue): three places read a
  "final weight" per expert as `W[-1, k]`, but `run_*_sleeping` only writes `W[t, i]` for experts awake at
  round t, so the last row is exactly 0 for anyone asleep in the final match -- 10 of 26 bookmakers here,
  PSC included. It hid the 4 worst frozen-advantage cases from `fixed_share.py`'s own diagnostic table and
  silently dropped PSC from `contextual_experts.py`'s per-league "most trusted" lists. Fixed via
  `sleeping_experts.last_awake_weight()`; NO prediction-quality metric changed (log-loss/Brier/RPS/accuracy
  in `results_table.csv`, `fixed_share_table.csv` and `contextual_experts_table.csv` verified identical
  before/after) -- the bug was purely in reading the learned state, not in the algorithms. See
  `docs/thesis_report.tex` sec. "Διόρθωση μέτρησης" and `docs/supervisor_report.tex`.
  Related, also fixed: `value_betting_online_kelly.py` printed its bootstrap table with the loss-based
  "negative = first-named is better" label, but that table's quantity is log-GROWTH (positive = better) --
  the printed sign convention was inverted (the CSV numbers and the stated conclusion were already right).

- **`bandit_oco.py`: the one-point bandit estimator learns NOTHING here, and the file exposed a trap
  that affects every other comparison in the project.** Two results.
  - **The trap, and the more important of the two.** Under Sleeping Experts the weight vector moves
    even at `eta = 0`, because the projection is re-done onto a different-dimensional simplex every
    time the awake set changes -- measured **0.69 away from uniform in max norm** with zero learning.
    So "beats the uniform average" is NOT by itself evidence that anything was learned. With an
    explicit `eta=0` control the uniform->OGD margin (0.001206) decomposes into **41% projection churn
    and the rest learning**. Any future Sleeping-Experts comparison against the uniform average should
    carry that control. It also agrees with the already-measured finding that late-period weight
    movement is dominated by the projection, not the gradient step.
  - **One-point bandit OCO learns nothing.** Train selection drives its step to ~1e-14 and the whole
    region 1e-14..9e-11 ties, i.e. the no-learning plateau. Quantified cause: single-draw alignment
    with the true gradient is **+0.002**, so the effective signal-to-noise over T rounds is ~0.55,
    below one -- the horizon is too short for a one-point estimator on a 25-dimensional simplex.
    **Mini-batching, the one fix the variance argument licenses, was tried** (4 batch sizes x 5 decades
    of step) and never produces an interior optimum. Two-point contributes a real 21.9%, but its shrink
    is driven to the lower bound and as delta->0 it converges to the exact directional derivative,
    i.e. it stops being bandit -- read it as an upper bound. Side finding: full information with the
    EXACT gradient ties with the shipped linearised OGD (+0.000002, p=0.934), so the linearisation
    costs nothing measurable.
- **`oco_kelly.py`: learning the Kelly MULTIPLIER works where learning the stake did not.** The earlier
  `value_betting_online_kelly.py` lost 9 of 20 with zero wins. The diagnosis is the COMPARATOR: a regret
  bound for a stake learner promises only against the best CONSTANT f, and the closed-form rule is not
  in that class since it produces a different f every round from `p_hat` and the odds, both known before
  the bet; it also throws away `p_hat`. Learning `lambda` in `f_t = lambda_t * f*_t` keeps the
  round-specific information and makes "best constant lambda" exactly the quantity `KELLY_MULT = 0.25`
  pins down. Result: **ties** fixed 1/4 (17 ties, 2 wins, 1 loss) and is ahead on bankroll on the
  profitable targets (WHC 6,494x vs 84x). The learned lambda tracks profitability -- 0.37 on the
  loss-making PSC, 0.79 on WHC -- which a constant cannot express. Two things to remember: the loss is
  **exp-concave in lambda**, so Online Newton Step gets O(log T) where OGD gets O(sqrt(T)), and **ONS
  selected a rate multiplier of exactly 1**, the only place in the project where the theory's a-priori
  constant needed no correction. Also `lambda_max * max(f*) < 1` is a SOLVENCY constraint, not a knob.
  Side finding: full Kelly ties in 16 of 20 cells, so the shipped 1/4 is more conservative than needed
  on profitable targets, with its risk concentrated almost entirely on PSC.
- **`regime_switching.py`: a Markov chain over latent MARKET states also fails, and the first failure
  is the instructive part.** The first version had the mean belief pinned at exactly **0.500** for both
  regimes: they never differentiate, because they see the same loss vector, so from a near-symmetric
  start they follow identical trajectories. It was therefore just OGD with the step divided by K, and
  lost significantly (+0.000128, p<0.001). The symmetric solution is a fixed point and needs an explicit
  differentiation mechanism. After adding belief sharpness, forgetting and a per-regime clock it **ties**
  (-0.000016, p=0.211) with beta=30 a genuine interior optimum (grid extended to 1000). But two things
  kill any claim the chain works: the held-out surface is **flat to 2e-6 across three decades of beta**
  while the train differences driving the selection are ten times larger, and **tau=0 is selected with
  belief sd = 0.0000**, so the belief freezes and what remains is a fixed blend of K OGD runs, i.e.
  ensembling. The latent regime does not correlate with the measured line-movement structure at all.
  Third independent instance of "once the step is tuned, moving mass over experts adds nothing".

## Known limitations (stated honestly in the LaTeX report, not hidden)

- The VC/BV rebrand merge is inferred from non-overlapping season presence, not confirmed by an official
  source -- more undetected rebrands are plausible, especially now with 22 leagues.
- Log-loss is the primary metric through most of the project; Brier is systematically analyzed only in the
  Shin de-vig comparison; RPS is computed everywhere but not systematically analyzed anywhere.
- Value-betting simulation ignores account limits/liquidity constraints. Worth noting this is not a
  detail but arguably the central economic result seen from the other side: the bookmakers the mixture
  CAN beat (WHC, IW -- soft, recreational-facing) are exactly the ones that close winning accounts,
  while the one that does not limit winners (Pinnacle) is the one that cannot be beaten. Access and
  exploitable edge are inversely related by the industry's own structure.
- A leave-one-bookmaker-out design can introduce look-ahead bias if the exclusion panel isn't restricted to
  information that would actually be available at bet-placement time -- one such case was found and fixed
  (the original BW value-betting test, see Key findings), but others may exist undetected in less obvious
  corners of the project.
- Causal explanations offered for some findings (e.g. why overround is rising) are plausible, not proven.

- **The deployment test is the strongest honest claim the project has** (`deployment_test.py`). Real
  calendar cut: fit on 65,009 rounds to 2024-12-31, score on 11,574 rounds of 2025-2026 (8,151 common).
  Against the bookmakers alive THERE, the mixture beats **every opening price significantly**
  (-0.0022 to -0.0065, all p<=0.007) and **ties every closing price** (B365C p=0.184, BWC p=0.334,
  BFEC p=0.051). Two readings are reported because they answer different questions: "online" (keeps
  learning through the test window, 1.00109) and "frozen" (weights locked at the cut, 1.00151) -- the
  online one wins consistently, so the right phrasing is "runs continuously", not "was trained until
  2024". Two things to say when quoting it: **Pinnacle is not in the main table** (coverage falls to
  59% after 2025, last quote 2026-01-14; forcing it in cuts the common set from 8.1k to 2.9k and leaves
  nothing significant anywhere), so it is tested one-to-one on its own rounds instead -- we beat PS
  significantly (-0.0022 to -0.0023, p 0.006-0.015) and tie PSC (p 0.149-0.549). And **the best
  forecaster in that window is not a bookmaker at all** but Betfair Exchange closing, a market price
  with no house margin.
- **Two claims about the mixture that were asserted and then MEASURED, one of which was wrong.**
  (a) "The mixture freezes late because the step decays as 1/sqrt(t)" -- FALSE. The step does fall 277x
  (10.24 -> 0.037), but mean per-round weight movement across the four quarters is 0.0226, 0.0156,
  0.0138, 0.0157: the last quarter moves as much as the second, and the weight vector shifts by L1 0.45
  within the tail (B365C goes 0.013 -> 0.262). Movement is dominated by the simplex PROJECTION each time
  the awake set changes, not by the gradient step. (b) Consequently the advantage of a long-trained
  mixture over one restarted inside the tail is the **start-up cost** being paid twice, not frozen
  accumulated knowledge.

## Open next steps (not yet built)

- A second sport (e.g. basketball) to test whether the OCO framework generalizes beyond football.
- A real Bayesian Model Averaging baseline for empirical comparison (discussed theoretically in the
  predecessor project's old progress report, never actually implemented). Note this baseline now has a
  second purpose beyond comparison: BMA needs a prior over models, OCO needs none, which is the sharpest
  technical illustration of the "no model, only track record" point the student wants to make (below).
- Systematic search for further bookmaker rebrand/merge cases across all 22 leagues.
- Investigating *why* overround is rising (would need data external to football-data.co.uk).
- **Re-tune the Hedge and FTRL step multipliers for the 26-expert panel.** The re-run puts their
  held-out minima at x0.05 and x0.005 against the shipped x0.25 and x0.05 (OGD's x100 is still right).
  Deliberately deferred because it means another full pipeline run plus another pass over the write-ups.
- A non-decaying constant step remains the only untested step-schedule idea -- everything else tried
  (Fixed-Share, joint step/calibration selection, per-expert clock, Markov transition kernels,
  MCGD-dependent-sampling schedule, AdaGrad, Bayesian Kelly, bandit gradient estimators, latent regime
  switching) has been tried and rejected.
- `oco_kelly.py` has been run only on the five CLOSING targets, via the closing-on-closing design.
  Extending the learned multiplier to the opening targets is open.
- The MCMC Kelly (`mcmc_kelly.py`) and Markov-kernel (`markov_ogd.py`) results are still not written
  into `thesis.tex`'s negative-results section -- they fit alongside Fixed-Share and the per-expert
  step clock. (The bandit, OCO-Kelly and regime-switching results ARE now written in, each with its own
  results section.)
- `docs/status_report.tex` still carries the pre-re-inclusion numbers throughout and has not been
  touched since; it needs the same pass `thesis.tex` just had.

## Write-up work discussed but not yet done

- **A short methodological passage on dialectical materialism**, requested by the student, ~2 paragraphs
  in the introduction or as a closing note of `thesis.tex`. Still not written -- checked directly, no
  occurrence of "διαλεκτ"/"materialism"/"Bitsakis"/"Engels" anywhere in the current file. The agreed shape: the framework holds
  no theory of how results are produced, only a record of what worked (praxis as the criterion), and
  every philosophical claim is anchored to a measurement made here -- learning pays only while the
  experts genuinely conflict (the heterogeneity result), the a-priori optimal step size was wrong by two
  orders of magnitude and the working one came from the data (the step tuning), and conclusions changed
  qualitatively when the sample grew quantitatively (5yr vs 10yr). Frame as Engels/Bitsakis/Levins-style
  dialectics-as-scientific-method, which is a POSITION in a live dispute (Lukács and the Frankfurt
  School reject extending dialectics to nature) -- say so rather than presenting it as neutral. The
  strongest version uses the framework to CRITIQUE the method too: materialism wants the move from
  appearance to essence, and OCO deliberately refuses to model mechanism.
- **The line-movement / information-arrival result** (`information_arrival.py`, the 26x gradient across
  line-movement quintiles) is in `status_report.tex` and in `thesis.tex`'s discussion chapter in summary
  form, but has **no dedicated results subsection with the actual table/figure in `thesis.tex` yet** --
  checked directly, `grep`ing for "information_arrival"/"26x"/"άφιξη πληροφορίας" in `thesis.tex` finds
  nothing. Add a proper §7.x with the quintile table before this counts as covered.
- **The material analysis of bookmaker business models** (Pinnacle's low-margin / winners-welcome model
  making its closing price a sharp-money aggregate rather than one firm's opinion, vs recreational books
  pricing to a biased public and therefore beatable in principle) is WRITTEN into `status_report.tex`
  ("Γιατί η Pinnacle δεν ξεπερνιέται") and into `thesis.tex`'s discussion chapter ("Η υλική βάση των
  διαφορών"), explicitly flagged as an industry claim rather than a result of this data. `thesis.tex`
  DOES have a `\begin{thebibliography}` with Kyle 1985, Glosten & Milgrom 1985, Shin 1992/1993, Levitt
  2004 as entries, but the body text everywhere uses author-year prose ("Kyle (1985)") with **zero actual
  `\cite{}` commands** -- checked directly (`grep '\\cite{'` finds nothing in the file). Wiring the prose
  mentions to `\cite{kyle1985}` etc. is a small, mechanical, not-yet-done task before this counts as
  properly cited. The passage also reframes the account-limits caveat (in "Γνωστοί περιορισμοί") as the
  same structure seen from the other side, which is the strongest form of that point.

## Where things live

- `data/raw/`, `data/processed/` -- inputs.
- `results/*.csv`, `results/*.png` -- every script's output; `docs/thesis.tex` embeds several of the PNGs
  (`\graphicspath{{../results/}}`) and quotes numbers from the CSVs, both by hand in the body and via the
  auto-generated appendix.
- **`docs/thesis.tex`** (+ compiled `thesis.pdf`) -- the current, canonical full Greek LaTeX thesis,
  ~60 pages: title/EL+EN abstract/TOC, related work, theory (with full proofs -- Hedge and OGD regret
  bounds, exp-concavity/Aggregating Algorithm, Kelly derivation, properness of log-loss), data,
  methodology, implementation (including the three internal bugs found, see CLAUDE.md), results
  (7 figures), discussion, conclusions/extensions, references, and an appendix of every result table.
  The appendix is generated by `scripts/make_appendix_tables.py` into `docs/appendix_tables.tex`, which
  `thesis.tex` `\input{}`s -- **rerun that script after any pipeline change**, before rebuilding, or the
  appendix goes stale silently. Replaces the older `thesis_report.tex` + `supervisor_report.tex` (moved
  to `docs/superseded/`, per explicit instruction to start one fresh thesis document and keep only
  `status_report.tex` from the old write-ups).
- `docs/status_report.tex` (+ `status_report.pdf`) -- the short status update for the supervisor, ~6
  pages, written largely in the student's own voice and kept deliberately separate from `thesis.tex`:
  Θέμα και δεδομένα / Τι έγινε / Αποτελέσματα (κατάταξη, panels, best configuration, no-Pinnacle,
  deployment test, value betting) / Συμπεράσματα, plus a footnote on the 5-season robustness check.
  **Deliberately carries no theory** -- no formulas, no method exposition; the supervisor knows it, and
  the student has asked repeatedly for less. Update this by hand whenever a headline number changes; it
  does NOT auto-regenerate the way the thesis appendix does.
- `docs/superseded/` -- `thesis_report.tex`, `supervisor_report.tex`, `progress_report.tex` (+ their
  PDFs), kept for reference only, no longer part of the active write-up. Don't copy numbers out of these
  without checking `results/` first -- several predate the step-multiplier correction and the IW
  exclusion.
- `docs/CODE_OVERVIEW.md` -- file/function index.
