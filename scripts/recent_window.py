"""
EVERY FORECASTING EXPERIMENT IN THE PROJECT, RERUN ON THE LAST FIVE SEASONS.

The headline numbers of this project are pooled over 10 seasons. That has one
concrete cost that only shows up when you look at WHICH bookmakers survive the
>=60% coverage floor: over 10 seasons only THREE closing columns do (PSC at
93.8%, B365C at 69.6%, BWC at 65.0%), because every other closing column only
starts around 2019/20. So the decade ranking still leans mostly on OPENING
prices, which the project itself shows are systematically worse.

Restrict to the last five seasons (2021/22-2025/26) and VC_BVC and WHC clear
the floor as well, giving five closing prices against five opening ones. The
comparison everyone actually cares about -- our mixture against the closing
market rather than a single column of it -- becomes possible.

The floor is deliberately the same number as final_ranking's, but note it is
not the same filter: inside a five-season window the series sit at either end
(a column starting in 2019/20 is near 100% here and near 0% over the decade),
so anything from 51% to 70% selects the identical panel. The results of this
file are not sensitive to the choice.

This file reruns, on that window and from scratch (fresh uniform-weight start,
not a slice of a 10-year run):
  A. dataset facts
  B. the Tier-A ranking            (cf. final_ranking.py)
  C. the significance battery      (cf. significance_test.py)
  D. panel x normalization grid    (cf. market_devig_comparisons.py)
  E. each algorithm vs the uniform average inside each panel
  F. the configuration stack       (cf. best_configuration.py)
  G. Pinnacle removed, two panels  (cf. no_pinnacle_panel.py)

The betting side lives in `recent_window_value_betting.py`.

DISCIPLINE, carried over from best_configuration.py: wherever series with
different coverage are ranked against each other, they are scored on ONE
common round set and split 75/25 chronologically. Coverage floors are
recomputed INSIDE the window -- a bookmaker's 10-season coverage says nothing
about its presence in the last five.

SIGN: diff = loss(A) - loss(B), so NEGATIVE means the first-named is better.

Produces:
  - results/recent_window_ranking.csv        (B)
  - results/recent_window_significance.csv   (C)
  - results/recent_window_panels.csv         (D, E)
  - results/recent_window_configs.csv        (F)
  - results/recent_window_no_pinnacle.csv    (G)
  - results/recent_window.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    DATA_DIR, OUT_DIR, EPS,
    per_round_expert_loss, evaluate,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping,
    load_universe_from, split_by_market_phase,
)
from calibration_analysis import _style_axes
from final_ranking import calibrate_full_history, RANK_COLORS
from significance_test import moving_block_bootstrap_test, paired_diff, full_length_raw_logloss

RECENT_SEASONS = 5
TRAIN_FRACTION = 0.75
COVERAGE_FLOOR = 60.0
CHALLENGER = "OGD"
DROPPED = ["PS", "PSC"]

ALGOS = ("Hedge", "OGD", "FTRL", "Uniform average")


# ------------------------------------------------------------------ helpers --

def slice_seasons(u, keep):
    panel, P, awake, y, dates, seasons = u
    m = np.isin(seasons, list(keep))
    return panel, P[m], awake[m], y[m], dates[m], seasons[m]


def fit_algorithms(P, awake, y, idx):
    """All four mixtures on a column subset. Returns {name: (phat, alive)}."""
    P_s, aw_s = P[:, idx, :], awake[:, idx]
    alive = aw_s.any(axis=1)
    loss = per_round_expert_loss(P_s, y)
    W = {
        "Hedge": run_hedge_sleeping(loss, aw_s, len(idx)),
        "OGD": run_ogd_sleeping(loss, aw_s, len(idx)),
        "FTRL": run_ftrl_sleeping(loss, aw_s, len(idx)),
        "Uniform average": np.divide(
            aw_s, aw_s.sum(axis=1, keepdims=True),
            out=np.zeros_like(aw_s, dtype=float),
            where=aw_s.sum(axis=1, keepdims=True) > 0),
    }
    return {n: (np.einsum("tn,tnk->tk", Wm, P_s), alive) for n, Wm in W.items()}


def per_round_series(probs, y, mask, T):
    """Raw and causally-calibrated per-round log-loss at full window length."""
    _, cal, yy, _ = calibrate_full_history(probs, y, mask)
    raw = full_length_raw_logloss(probs, y, T, mask=mask)
    calib = np.full(T, np.nan)
    calib[mask] = -np.log(np.clip(cal[np.arange(len(yy)), yy], EPS, 1.0))
    return raw, calib


def common_split(series_dicts, T):
    """Intersection where every series is defined, plus a 75/25 chronological
    split of it. Returns (common, train, val, n_common)."""
    common = np.ones(T, dtype=bool)
    for s in series_dicts:
        common &= ~np.isnan(s["raw"]) & ~np.isnan(s["calibrated"])
    n = int(common.sum())
    ic = np.flatnonzero(common)
    k = int(round(n * TRAIN_FRACTION))
    tr = np.zeros(T, bool); tr[ic[:k]] = True
    va = np.zeros(T, bool); va[ic[k:]] = True
    return common, tr, va, n


def paired(a_series, b_series, common, stage):
    a = np.where(common, a_series[stage], np.nan)
    b = np.where(common, b_series[stage], np.nan)
    d, n = paired_diff(a, b)
    r = moving_block_bootstrap_test(d)
    return n, r


def head(title):
    print(f"\n{'#' * 74}\n{title}\n{'#' * 74}")


# ------------------------------------------------- B. the Tier-A ranking --

def section_ranking(panel, P, awake, y, seasons):
    head("B. Κατάταξη Tier-A (πλήρες panel, απλή κανονικοποίηση)")
    T, N, _ = P.shape
    fits = fit_algorithms(P, awake, y, list(range(N)))
    coverage = awake.mean(axis=0) * 100

    entries = {n: (p, m, "algorithm", "---") for n, (p, m) in fits.items()}
    for k, book in enumerate(panel):
        if coverage[k] >= COVERAGE_FLOOR and len(set(seasons[awake[:, k]])) > 1:
            entries[book] = (P[:, k, :], awake[:, k],
                             "bookmaker", "closing" if book.endswith("C") else "opening")

    rows = []
    for name, (probs, mask, kind, phase) in entries.items():
        p, cal, yy, _ = calibrate_full_history(probs, y, mask)
        raw_m, cal_m = evaluate(p, yy), evaluate(cal, yy)
        rows.append({
            "name": name, "type": kind, "phase": phase,
            "coverage_pct": mask.mean() * 100, "rounds": int(mask.sum()),
            "raw_log_loss": raw_m["log_loss"], "calibrated_log_loss": cal_m["log_loss"],
            "raw_brier": raw_m["brier"], "calibrated_brier": cal_m["brier"],
            "calibrated_accuracy": cal_m["accuracy"],
        })
    table = pd.DataFrame(rows).sort_values("calibrated_log_loss").reset_index(drop=True)
    pd.set_option("display.width", 210)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    books = table[table["type"] == "bookmaker"]
    nc = int((books["phase"] == "closing").sum())
    print(f"\n{len(books)} bookmakers περνούν το {COVERAGE_FLOOR:.0f}% "
          f"({nc} closing / {len(books) - nc} opening)")
    ll = list(table.sort_values("calibrated_log_loss")["name"])
    br = list(table.sort_values("calibrated_brier")["name"])
    print(f"log-loss και Brier δίνουν ίδια σειρά: {ll == br}")
    if ll != br:
        print(f"  log-loss: {ll}\n  Brier   : {br}")
    return table


# ------------------------------------------ C. the significance battery --

def section_significance(panel, P, awake, y):
    head("C. Στατιστική σημαντικότητα")
    T, N, _ = P.shape
    fits = fit_algorithms(P, awake, y, list(range(N)))

    ser = {}
    for name, (probs, mask) in fits.items():
        raw, cal = per_round_series(probs, y, mask, T)
        ser[name] = {"raw": raw, "calibrated": cal}
    for book in ("PSC", "B365C"):
        k = panel.index(book)
        raw, cal = per_round_series(P[:, k, :], y, awake[:, k], T)
        ser[book] = {"raw": raw, "calibrated": cal}

    pairs = [("OGD", "PSC", "raw"), ("OGD", "PSC", "calibrated"),
             ("OGD", "B365C", "raw"),
             ("OGD", "Hedge", "raw"), ("OGD", "FTRL", "raw"),
             ("OGD", "Uniform average", "raw"),
             ("Hedge", "Uniform average", "raw"), ("FTRL", "Uniform average", "raw")]
    rows = []
    for a, b, stage in pairs:
        common, _, _, _ = common_split([ser[a], ser[b]], T)
        n, r = paired(ser[a], ser[b], common, stage)
        rows.append({"comparison": f"{a} vs {b} ({stage})", "n": n, **r})
    for a in ALGOS:
        common = ~np.isnan(ser[a]["raw"]) & ~np.isnan(ser[a]["calibrated"])
        n, r = paired({"x": np.where(common, ser[a]["raw"], np.nan)},
                      {"x": np.where(common, ser[a]["calibrated"], np.nan)}, common, "x")
        rows.append({"comparison": f"{a}: raw vs calibrated", "n": n, **r})

    sig = pd.DataFrame(rows)
    print(sig.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    return sig


# --------------------------------- D/E. panels x normalizations, vs uniform --

def section_panels(universes):
    head("D. Panel x κανονικοποίηση  +  E. κάθε αλγόριθμος vs απλός μέσος όρος")
    rows, vs_uniform = [], []
    for norm, u in universes.items():
        panel, P, awake, y, dates, seasons = u
        T, N, _ = P.shape
        open_idx, close_idx = split_by_market_phase(panel)
        panels = {"opening": open_idx, "full": list(range(N)), "closing": close_idx}

        ser = {}
        for pname, idx in panels.items():
            for aname, (probs, mask) in fit_algorithms(P, awake, y, idx).items():
                raw, cal = per_round_series(probs, y, mask, T)
                ser[(pname, aname)] = {"raw": raw, "calibrated": cal}

        # one common round set across all three panels, so closing<full<opening
        # is not an artifact of each panel being scored on its own matches
        common, _, _, n_common = common_split(list(ser.values()), T)
        print(f"\n[{norm}] κοινοί γύροι και στα τρία panel: {n_common}")
        for (pname, aname), s in ser.items():
            rows.append({"normalization": norm, "panel": pname, "algorithm": aname,
                         "common_rounds": n_common,
                         "log_loss": float(np.nanmean(np.where(common, s["raw"], np.nan))),
                         "calibrated_log_loss": float(np.nanmean(np.where(common, s["calibrated"], np.nan)))})
        for pname in panels:
            for aname in ("Hedge", "OGD", "FTRL"):
                n, r = paired(ser[(pname, aname)], ser[(pname, "Uniform average")], common, "raw")
                verdict = ("ισοπαλία" if not r["significant_95"]
                           else (f"{aname} wins" if r["mean_diff"] < 0 else "Uniform wins"))
                vs_uniform.append({"normalization": norm, "panel": pname,
                                   "algorithm": aname, "n": n, "verdict": verdict, **r})

    grid = pd.DataFrame(rows)
    piv = grid.pivot_table(index="algorithm", columns=["panel", "normalization"],
                           values="log_loss")
    piv = piv.reindex(columns=pd.MultiIndex.from_product(
        [["opening", "full", "closing"], ["basic", "shin"]]))
    print("\n--- raw log-loss, ίδιοι γύροι σε κάθε κελί ---")
    print(piv.reindex(list(ALGOS)).to_string(float_format=lambda v: f"{v:.5f}"))

    vu = pd.DataFrame(vs_uniform)
    print("\n--- κάθε αλγόριθμος vs απλός μέσος όρος, μέσα στο ίδιο panel ---")
    print(vu[["normalization", "panel", "algorithm", "mean_diff", "p_value", "verdict"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    lost = vu[vu["verdict"] == "Uniform wins"]
    print(f"\nΟ απλός μέσος όρος κερδίζει σε {len(lost)} από {len(vu)} τεστ "
          f"({sorted(set(lost['algorithm']))} στα {sorted(set(lost['panel']))})")

    # ordering check
    ok = True
    for norm in universes:
        for a in ALGOS:
            g = grid[(grid.normalization == norm) & (grid.algorithm == a)].set_index("panel")
            ok &= g.loc["closing", "log_loss"] < g.loc["full", "log_loss"] < g.loc["opening", "log_loss"]
    print(f"closing < full < opening παντού: {ok}")
    return grid, vu


# ------------------------------------------- F. the configuration stack --

def section_configs(universes):
    head("F. Στοίβαξη διαμορφώσεων (vs PSC και vs πλήρες panel)")
    panel, P_b, awake, y, dates, seasons = universes["basic"]
    _, P_s, _, _, _, _ = universes["shin"]
    T, N, _ = P_b.shape
    open_idx, close_idx = split_by_market_phase(panel)
    single = [k for k in range(N) if len(set(seasons[awake[:, k]])) <= 1]
    close_multi = [k for k in close_idx if k not in single]

    configs = {
        "πλήρες panel, απλή":       (P_b, list(range(N))),
        "closing-only, απλή":       (P_b, close_idx),
        "πλήρες panel, Shin":       (P_s, list(range(N))),
        "closing-only + Shin":      (P_s, close_idx),
        "closing+Shin χωρίς μονής σεζόν": (P_s, close_multi),
    }
    ser = {}
    for name, (Pu, idx) in configs.items():
        probs, mask = fit_algorithms(Pu, awake, y, idx)["OGD"]
        raw, cal = per_round_series(probs, y, mask, T)
        ser[name] = {"raw": raw, "calibrated": cal}
        print(f"  {name:32s} {len(idx):2d} experts, {int(mask.sum())} γύροι")

    k = panel.index("PSC")
    raw, cal = per_round_series(P_b[:, k, :], y, awake[:, k], T)
    ser["PSC μόνος του"] = {"raw": raw, "calibrated": cal}

    common, tr, va, n_common = common_split(list(ser.values()), T)
    print(f"\nίδιοι {n_common} γύροι για κάθε γραμμή "
          f"(train {int(tr.sum())} / held-out {int(va.sum())})\n")

    rows = []
    for name, s in ser.items():
        rows.append({"config": name,
                     "full": float(np.nanmean(np.where(common, s["calibrated"], np.nan))),
                     "train": float(np.nanmean(np.where(tr, s["calibrated"], np.nan))),
                     "held_out": float(np.nanmean(np.where(va, s["calibrated"], np.nan)))})
    table = pd.DataFrame(rows).sort_values("full").reset_index(drop=True)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n--- ζευγαρωτά, βαθμονομημένο ---")
    sig_rows = []
    for name in ser:
        for ref in ("PSC μόνος του", "πλήρες panel, απλή"):
            if name == ref or name == "PSC μόνος του":
                continue
            n, r = paired(ser[name], ser[ref], common, "calibrated")
            verdict = ("ισοπαλία" if not r["significant_95"]
                       else (f"{name} better" if r["mean_diff"] < 0 else f"{ref} better"))
            sig_rows.append({"config": name, "vs": ref, "n": n, "verdict": verdict, **r})
    sig = pd.DataFrame(sig_rows)
    print(sig[["config", "vs", "mean_diff", "ci_low", "ci_high", "p_value", "verdict"]]
          .to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    return table, sig


# ------------------------------------------------- G. Pinnacle removed --

def section_no_pinnacle(universes):
    head("G. Χωρίς Pinnacle (PS, PSC), δύο panel")
    panel, P, awake, y, dates, seasons = universes["basic"]
    T, N, _ = P.shape
    open_idx, close_idx = split_by_market_phase(panel)

    out_tables, out_sigs = [], []
    for label, base in (("πλήρες panel", list(range(N))), ("closing-only panel", close_idx)):
        idx = [k for k in base if panel[k] not in DROPPED]
        print(f"\n--- {label}: {len(idx)} experts ---")
        fits = fit_algorithms(P, awake, y, idx)
        coverage = awake.mean(axis=0) * 100
        entries = {n: (p, m, "algorithm", "---") for n, (p, m) in fits.items()}
        for k in base:
            book = panel[k]
            if book in DROPPED:
                continue
            if coverage[k] >= COVERAGE_FLOOR and len(set(seasons[awake[:, k]])) > 1:
                entries[book] = (P[:, k, :], awake[:, k], "bookmaker",
                                 "closing" if book.endswith("C") else "opening")
        ser = {}
        for name, (probs, mask, kind, phase) in entries.items():
            raw, cal = per_round_series(probs, y, mask, T)
            ser[name] = {"raw": raw, "calibrated": cal, "type": kind, "phase": phase}

        common, tr, va, n_common = common_split(list(ser.values()), T)
        print(f"κοινοί γύροι: {n_common}")
        rows = [{"panel": label, "name": n, "type": s["type"], "phase": s["phase"],
                 "calibrated_log_loss": float(np.nanmean(np.where(common, s["calibrated"], np.nan))),
                 "held_out": float(np.nanmean(np.where(va, s["calibrated"], np.nan)))}
                for n, s in ser.items()]
        t = pd.DataFrame(rows).sort_values("calibrated_log_loss").reset_index(drop=True)
        t.insert(0, "rank", np.arange(1, len(t) + 1))
        print(t.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

        sig_rows = []
        for name, s in ser.items():
            if s["type"] != "bookmaker":
                continue
            n, r = paired(ser[CHALLENGER], s, common, "calibrated")
            verdict = ("ισοπαλία" if not r["significant_95"]
                       else (f"{CHALLENGER} better" if r["mean_diff"] < 0 else f"{name} better"))
            sig_rows.append({"panel": label, "bookmaker": name, "phase": s["phase"],
                             "n": n, "verdict": verdict, **r})
        sg = pd.DataFrame(sig_rows).sort_values("mean_diff")
        print(sg[["bookmaker", "phase", "mean_diff", "p_value", "verdict"]]
              .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        wins = int((sg["verdict"] == f"{CHALLENGER} better").sum())
        losses = int((sg["verdict"].str.endswith("better") &
                      (sg["verdict"] != f"{CHALLENGER} better")).sum())
        print(f"{CHALLENGER}: {wins} νίκες, {losses} ήττες, "
              f"{len(sg) - wins - losses} ισοπαλίες, από {len(sg)}")
        out_tables.append(t)
        out_sigs.append(sg)
    return pd.concat(out_tables, ignore_index=True), pd.concat(out_sigs, ignore_index=True)


# -------------------------------------- H. against the CURRENT market only --

def section_live_market(u_full, tail_fraction=0.25):
    """The economically meaningful comparison: our mixture against the
    bookmakers a bettor can ACTUALLY use today.

    Everything else in the project ranks against a Tier-A set defined by
    10-season coverage, which is a historical accident -- a bookmaker that
    quoted heavily until 2024 and then vanished still clears the floor. Here
    the floor is recomputed on the chronological tail alone, so only series
    that are alive in the recent market qualify. The algorithms are still
    fitted on the FULL history (one continuous causal run, no restart); only
    the scoring window is the tail.
    """
    panel, P, awake, y, dates, seasons = u_full
    T, N, _ = P.shape
    d = pd.to_datetime(pd.Series(dates))
    cut = int(round(T * (1 - tail_fraction)))
    tail = np.zeros(T, bool); tail[cut:] = True
    head(f"H. Μόνο η σημερινή αγορά: τελευταίο {tail_fraction:.0%} "
         f"({str(d[cut])[:10]} -> {str(d[T-1])[:10]})")

    ser = {}
    for name, (probs, mask) in fit_algorithms(P, awake, y, list(range(N))).items():
        _, cal = per_round_series(probs, y, mask, T)
        ser[name] = {"calibrated": cal, "type": "algorithm", "phase": "---", "cov": 100.0}

    cov_tail = awake[tail].mean(axis=0) * 100
    for k, b in enumerate(panel):
        if cov_tail[k] >= COVERAGE_FLOOR and len(set(seasons[awake[:, k]])) > 1:
            _, cal = per_round_series(P[:, k, :], y, awake[:, k], T)
            ser[b] = {"calibrated": cal, "type": "bookmaker", "cov": cov_tail[k],
                      "phase": "closing" if b.endswith("C") else "opening"}

    defined = np.ones(T, bool)
    for s in ser.values():
        defined &= ~np.isnan(s["calibrated"])
    sel = tail & defined
    books = [k for k, s in ser.items() if s["type"] == "bookmaker"]
    print(f"ζωντανοί bookmakers ({COVERAGE_FLOOR:.0f}% μέσα στην ουρά): {books}")
    print(f"κοινοί γύροι: {int(sel.sum())} από {int(tail.sum())}\n")

    rows = [{"name": k, "type": s["type"], "phase": s["phase"],
             "coverage_tail_pct": s["cov"],
             "log_loss": float(np.nanmean(np.where(sel, s["calibrated"], np.nan)))}
            for k, s in ser.items()]
    t = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    print(t.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    sig = []
    for name, s in ser.items():
        if s["type"] != "bookmaker":
            continue
        _, r = paired(ser[CHALLENGER], s, sel, "calibrated")
        v = ("ισοπαλία" if not r["significant_95"]
             else (f"{CHALLENGER} better" if r["mean_diff"] < 0 else f"{name} better"))
        sig.append({"bookmaker": name, "phase": s["phase"], "verdict": v, **r})
    sg = pd.DataFrame(sig).sort_values("mean_diff")
    print(f"\n{CHALLENGER} ζευγαρωτά (αρνητικό = {CHALLENGER} καλύτερο):")
    print(sg[["bookmaker", "phase", "mean_diff", "ci_low", "ci_high", "p_value", "verdict"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    w = int((sg["verdict"] == f"{CHALLENGER} better").sum())
    l = int((sg["verdict"].str.endswith("better") & (sg["verdict"] != f"{CHALLENGER} better")).sum())
    print(f"-> {CHALLENGER}: {w} νίκες, {l} ήττες, {len(sg) - w - l} ισοπαλίες, "
          f"θέση #{int(t[t['name'] == CHALLENGER]['rank'].iloc[0])} από {len(t)}")
    return t, sg


# ------------------------------------------------------------------- plot --

def plot_ranking(table, out_path):
    df = table.sort_values("calibrated_log_loss").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.46 * len(df) + 2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ypos = np.arange(len(df))[::-1]
    for yp, (_, r) in zip(ypos, df.iterrows()):
        c = RANK_COLORS[r["type"]]
        ax.plot([r["raw_log_loss"], r["calibrated_log_loss"]], [yp, yp],
                color=c, linewidth=1.6, zorder=2, alpha=0.9)
        ax.scatter([r["raw_log_loss"]], [yp], s=64, facecolor="#fcfcfb",
                   edgecolor=c, linewidth=1.7, zorder=3)
        ax.scatter([r["calibrated_log_loss"]], [yp], s=64, facecolor=c,
                   edgecolor=c, linewidth=1.0, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([n + ("" if p == "---" else f"  ({p})")
                        for n, p in zip(df["name"], df["phase"])],
                       fontsize=9, color="#0b0b0b")
    ax.set_xlabel("Log-loss", color="#52514e")
    ax.set_title(f"Τελευταίες {RECENT_SEASONS} σεζόν, πλήρες panel\n"
                 "κενό = raw, γεμάτο = βαθμονομημένο",
                 color="#0b0b0b", fontsize=12, pad=12)
    _style_axes(ax)
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


# ------------------------------------------------------------------- main --

def main():
    basic = load_universe_from(os.path.join(DATA_DIR, "odds_long.csv"))
    shin = load_universe_from(os.path.join(DATA_DIR, "odds_long_shin.csv"))
    assert np.array_equal(basic[3], shin[3]), "τα δύο datasets πρέπει να ευθυγραμμίζονται"

    all_seasons = sorted(set(basic[5]))
    keep = all_seasons[-RECENT_SEASONS:]
    universes = {"basic": slice_seasons(basic, keep), "shin": slice_seasons(shin, keep)}
    panel, P, awake, y, dates, seasons = universes["basic"]

    head(f"A. Δεδομένα: τελευταίες {RECENT_SEASONS} σεζόν")
    cov = awake.mean(axis=0) * 100
    alive = [k for k in range(len(panel)) if awake[:, k].any()]
    print(f"σεζόν: {keep}")
    print(f"γύροι: {len(y)}  (από {len(basic[3])} στις 10 σεζόν)")
    print(f"experts με >=1 απόδοση: {len(alive)} από {len(panel)}")
    print(f"κάλυψη: {cov[alive].min():.1f}% ({panel[alive[int(np.argmin(cov[alive]))]]}) "
          f"ως {cov[alive].max():.1f}% ({panel[alive[int(np.argmax(cov[alive]))]]})")
    print(f"πάνω από το {COVERAGE_FLOOR:.0f}%: "
          f"{[panel[k] for k in range(len(panel)) if cov[k] >= COVERAGE_FLOOR]}")

    rank = section_ranking(panel, P, awake, y, seasons)
    sig = section_significance(panel, P, awake, y)
    grid, vu = section_panels(universes)
    cfg, cfg_sig = section_configs(universes)
    npt, npsig = section_no_pinnacle(universes)
    live, live_sig = section_live_market(basic)

    for df, name in ((pd.concat([live, live_sig], ignore_index=True), "recent_window_live_market"),
                     (rank, "recent_window_ranking"),
                     (sig, "recent_window_significance"),
                     (pd.concat([grid, vu], ignore_index=True), "recent_window_panels"),
                     (pd.concat([cfg, cfg_sig], ignore_index=True), "recent_window_configs"),
                     (pd.concat([npt, npsig], ignore_index=True), "recent_window_no_pinnacle")):
        p = os.path.join(OUT_DIR, f"{name}.csv")
        df.to_csv(p, index=False)
        print(f"Saved: {p}")
    plot_ranking(rank, os.path.join(OUT_DIR, "recent_window.png"))


if __name__ == "__main__":
    main()
