# Online Convex Optimization based tuning of Mixture of Experts for Sports Outcome Prediction

Diploma thesis, School of Electrical and Computer Engineering, Technical University of Crete.

Each betting operator is treated as an **expert** quoting odds on the three outcomes of a football
match (home / draw / away), from which probabilities are derived. The task is to learn a weight
vector online, match by match, so that the mixture of experts outperforms each of them
individually. Nothing is fitted retrospectively: every prediction is out of sample by construction.

The dataset covers **22 leagues, 10 seasons and 76,584 matches**, with **26 bookmakers** as
experts (13 opening prices, 13 closing).

## Three OCO stages

1. **Weighting.** Hedge, Online Gradient Descent and Follow-The-Regularized-Leader, generalised
   through the *Sleeping Experts* reduction so that operators who quote only part of the fixture
   list can still take part: each expert keeps its own persistent state, the prediction is formed
   from the awake subset alone, and a sleeping expert's state is frozen rather than decayed.
2. **Calibration.** Online temperature scaling on top of the mixture, which is a second OCO
   problem: the loss is convex in the exponent, so the temperature is learnable by projected OGD
   and the whole chain stays causal.
3. **Stake size.** The Kelly criterion, with the *multiplier* learned online by Online Newton Step
   rather than pinned to a constant. What is learned matters: learning the stake directly fails,
   because the comparator "best constant stake" does not contain the closed-form rule at all.

Every comparison is tested with a moving block bootstrap, since consecutive match losses are not
independent.

## Headline results

- The mixture significantly beats the uniform average and almost every individual bookmaker.
- It draws level with the sharpest bookmaker in the configuration finally selected. The one series
  that significantly beats it is **Betfair Exchange**, which is not a bookmaker but an exchange,
  that is a market price with no operator margin.
- **Better prediction does not mean profit.** The quality gained is smaller than the market's
  margin, and whatever edge exists is concentrated against specific, recreational-facing houses.
- **41% of the margin over the uniform average is not learning at all.** Under the Sleeping Experts
  reduction the weight vector moves even with a zero step, because the projection is re-done onto a
  simplex of different dimension whenever the awake set changes. A zero-step control separates the
  two; any comparison against a uniform baseline should carry one.
- Learned weighting pays off only when the experts differ in *how they are produced*. On a
  homogeneous panel the plain average is at least as robust.

## Running it

Dependencies are `numpy`, `pandas`, `matplotlib`:

```
pip install -r requirements.txt
```

The raw data is not committed (third-party, and the processed tensors exceed GitHub's file size
limit). Fetch and build it first:

```
cd scripts
pwsh ./download_data.ps1          # football-data.co.uk, Main Leagues
python run_pipeline.py --with-data
```

After that the raw data is in place and the pipeline runs on its own:

```
python run_pipeline.py            # all 34 steps, in dependency order
python run_pipeline.py --list     # show the plan without running it
python run_pipeline.py --only learning_rate_study
python run_pipeline.py --from market_devig_comparisons
```

A full run takes about **85 minutes**, dominated by the MCMC and grid-sweep experiments rather
than by anything repeated. Results are written to `results/` as flat CSV and PNG.

**Scripts must be run from inside `scripts/`.** Paths are relative to it.

### Checking a change did not move anything

The project's rule for touching shared code is: snapshot the outputs, rerun, prove they are
unchanged. That is automated:

```
python check_results.py save ../results_baseline
# ... make the change, rerun the pipeline ...
python check_results.py diff ../results_baseline
```

CSVs are compared on values (row order can legitimately differ), everything else by bytes. The
pipeline is deterministic: every script that uses randomness seeds it explicitly.

## Layout

```
scripts/          the pipeline, one script per question
  sleeping_experts.py     shared library: loading, the three algorithms, metrics
  run_pipeline.py         runs everything in dependency order
  check_results.py        snapshot and compare results/
  deprecated/             superseded scripts, kept for reference
data/raw/         football-data.co.uk CSVs          (not committed)
data/processed/   the long-format odds table        (not committed, >100 MB)
results/          every CSV and figure the thesis quotes
docs/             the thesis (Greek and English), XeLaTeX
```

`docs/thesis.tex` is the Greek edition and `docs/thesis_en_v2.tex` the English one. Both need
**XeLaTeX** (fontspec and polyglossia for the Greek); pdfLaTeX will not build them.

## Notes on the data

`VC` and `BV` are the same company either side of a rebrand and are merged into one continuous
expert, inferred from their non-overlapping presence rather than from an official source. The
aggregate columns (`Max`, `Avg`, Betbrain) are excluded: they are functions of the other columns,
not independent experts. A bookmaker name ending in `C` is the closing-odds counterpart of the
identically-prefixed opening one, which is football-data.co.uk's own convention.

Odds data is published by [football-data.co.uk](https://www.football-data.co.uk/) and is not
redistributed here.
