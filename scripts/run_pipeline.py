"""
RUN THE WHOLE PIPELINE, IN DEPENDENCY ORDER.

Until now the order lived only in prose in CLAUDE.md, and running everything
meant invoking three dozen scripts by hand in the right sequence. This file is
that sequence, executable.

Each step is a separate process, which is deliberate: the scripts were written
to be standalone and several of them rely on starting from a clean interpreter.
The cost of a process launch is irrelevant next to what they compute.

Three steps read another step's CSV rather than only the processed data, so
order is load-bearing for them:
  - market_devig_comparisons and value_betting read results_table.csv
  - value_betting_ogd_vs_ftrl and value_betting_circular read
    value_betting_all_bookmakers_table.csv (the first of the two asserts its own
    OGD half reproduces it, so a stale file turns into a confusing failure)

data_processer only needs re-running when the raw downloads change, so it is
opt-in via --with-data. make_appendix_tables is not a dependency of anything but
goes stale silently, so it runs last, in both languages.

Usage:
  python run_pipeline.py                 run everything except data processing
  python run_pipeline.py --list          print the plan and exit
  python run_pipeline.py --with-data     include data_processer first
  python run_pipeline.py --only NAME...  run just these steps
  python run_pipeline.py --from NAME     run from this step onwards
  python run_pipeline.py --skip NAME...  run everything but these
  python run_pipeline.py --keep-going    do not stop at the first failure
"""

import argparse
import os
import subprocess
import sys
import time

# (name, note). Order is the dependency order; everything in a later position
# may depend on anything earlier, never the other way round.
STEPS = [
    ("sleeping_experts_experiment", "the three algorithms over the full panel"),
    ("calibration_analysis",        "reliability of the raw mixture"),
    ("calibration_correction",      "online temperature scaling"),
    ("final_ranking",               "the reference ranking"),

    ("significance_test",           "moving block bootstrap battery"),
    ("fixed_share",                 "Fixed-Share sweep over alpha"),

    ("value_betting",               "Kelly simulations; writes the bookmaker sweep"),
    ("value_betting_online_kelly",  "the earlier stake learner, kept for contrast"),
    ("value_betting_ogd_vs_ftrl",   "asserts it reproduces the OGD half"),
    ("value_betting_circular",      "leakage control"),
    ("value_betting_sharp_anchor",  "anchor-bookmaker variant"),
    ("value_betting_by_band",       "profit by probability band"),

    ("contextual_experts",          "per-league specialisation"),
    ("outcome_class_accuracy",      "performance by H/D/A"),
    ("outcome_class_comparison",    "across methods"),
    ("outcome_class_accuracy_test", "significance of the above"),

    ("temporal_trends",             "Mann-Kendall over seasons"),
    ("pooling_window_comparison",   "time-window comparison"),
    ("calibration_per_season",      "calibration fitted per season"),
    ("calibration_warmstart",       "with a starting temperature"),

    ("market_devig_comparisons",    "opening/closing panels, Shin vs proportional"),
    ("information_arrival",         "closing advantage by line movement"),
    ("best_configuration",          "stacking the configurations"),
    ("no_pinnacle_panel",           "standalone: drops PS/PSC and refits"),
    ("recent_window",               "the last five seasons"),
    ("recent_window_value_betting", "value betting inside that window"),

    ("deployment_test",             "real date cut, 2025-26"),
    ("accuracy_by_band",            "accuracy by probability band"),
    ("learning_rate_study",         "step multiplier and constant-step sweeps"),
    ("markov_ogd",                  "transition kernels on the weight vector"),
    ("mcmc_kelly",                  "posterior-expected Kelly"),
    ("bandit_oco",                  "one- and two-point gradient estimators"),
    ("oco_kelly",                   "learned Kelly multiplier"),
    ("regime_switching",            "latent market regimes"),
]

DATA_STEP = ("data_processer", "rebuilds odds_long.csv from data/raw")
APPENDIX = [
    ("make_appendix_tables", "the thesis appendix, Greek", []),
    ("make_appendix_tables", "the thesis appendix, English", ["--en"]),
]


def run(name, args, env):
    cmd = [sys.executable, f"{name}.py", *args]
    t0 = time.time()
    r = subprocess.run(cmd, env=env)
    return r.returncode, time.time() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true")
    p.add_argument("--with-data", action="store_true")
    p.add_argument("--only", nargs="+", metavar="NAME")
    p.add_argument("--from", dest="start", metavar="NAME")
    p.add_argument("--skip", nargs="+", default=[], metavar="NAME")
    p.add_argument("--keep-going", action="store_true")
    a = p.parse_args()

    plan = ([DATA_STEP] if a.with_data else []) + STEPS
    names = [n for n, _ in plan]

    if a.only:
        unknown = [n for n in a.only if n not in names + [APPENDIX[0][0]]]
        if unknown:
            sys.exit(f"unknown step(s): {', '.join(unknown)}")
        plan = [s for s in plan if s[0] in a.only]
    elif a.start:
        if a.start not in names:
            sys.exit(f"unknown step: {a.start}")
        plan = plan[names.index(a.start):]
    plan = [s for s in plan if s[0] not in a.skip]

    appendix = [] if (a.only or a.skip) else APPENDIX

    if a.list:
        for i, (n, note) in enumerate(plan, 1):
            print(f"{i:3}. {n:30} {note}")
        for n, note, args in appendix:
            print(f"   . {n:30} {note}")
        return

    # the scripts resolve their paths relative to this directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    results, t_all = [], time.time()
    for i, (name, note) in enumerate(plan, 1):
        print(f"\n{'=' * 72}\n[{i}/{len(plan)}] {name}  --  {note}\n{'=' * 72}",
              flush=True)
        code, dt = run(name, [], env)
        results.append((name, code, dt))
        if code and not a.keep_going:
            print(f"\n{name} failed with exit code {code}; stopping.")
            break
    else:
        for name, note, args in appendix:
            label = f"{name} {' '.join(args)}".strip()
            print(f"\n{'=' * 72}\n[appendix] {label}  --  {note}\n{'=' * 72}",
                  flush=True)
            code, dt = run(name, args, env)
            results.append((label, code, dt))

    print(f"\n{'=' * 72}\nSUMMARY  ({time.time() - t_all:.0f}s total)\n{'=' * 72}")
    for name, code, dt in sorted(results, key=lambda r: -r[2]):
        print(f"  {'ok  ' if code == 0 else f'FAIL{code}'}  {dt:7.1f}s  {name}")
    failed = [n for n, c, _ in results if c]
    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
