"""
DEPLOYMENT TEST: fit on 2016-2024, predict 2025+, is the prediction better?

Every other ranking in this project scores the mixture over the same history
it learned on. That is legitimate for OCO -- the algorithm is causal, so each
prediction already only uses the past, and cumulative loss over the sequence
is exactly what a regret bound is about. It is not, however, the question a
supervisor or a bettor actually asks, which is: *take what it knew at the end
of 2024, run it through 2025, and see whether it beats the market that exists
then.*

That is what this file does. The cut is a real calendar date, not a fraction
of the series, and the opponents are recomputed from coverage INSIDE the test
window -- which matters more than it sounds: Pinnacle, the benchmark for the
whole project, drops to 59% coverage after 2025 and stops quoting altogether
in January 2026. The sharpest thing left in the data is the Betfair Exchange
closing price (96% coverage), which is a market price rather than a
bookmaker's quote.

TWO READINGS of "trained on 2016-2024", both reported because they answer
different questions:

  ONLINE    the mixture keeps updating through the test window as results come
            in. Still causal, and what you would actually deploy.
  FROZEN    weights are locked at the cut and applied unchanged. The strict
            train-once reading. Weights are still projected onto each round's
            awake subset -- that is required to produce a valid prediction at
            all, not a learning step.

The calibrator matches the variant: its learning rate is always selected on
TRAIN rounds only, then either kept live or frozen at the cut temperature.

SIGN: diff = loss(mixture) - loss(bookmaker), so NEGATIVE means we are better.

Produces:
  - results/deployment_test.csv
  - results/deployment_test_significance.csv
  - results/deployment_test.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    DATA_DIR, OUT_DIR, EPS, M,
    per_round_expert_loss, project_to_simplex, OGD_STEP_MULTIPLIER,
    load_universe_from, split_by_market_phase,
)
from calibration_analysis import _style_axes
from final_ranking import RANK_COLORS
from calibration_correction import temperature_calibrate, select_lr_on_train, LR_GRID
from significance_test import moving_block_bootstrap_test, paired_diff

CUT = np.datetime64("2025-01-01")

COVERAGE_FLOOR = 70.0
"""Recomputed inside the TEST window. Everything above it goes in the table and
is scored on one shared set of matches, which is what makes the rows
comparable."""

ASIDE = ("PS", "PSC")
"""Pinnacle is reported separately, one-to-one, on the rounds where it still
quotes. It falls to ~59% coverage after 2025 and stops on 2026-01-14, so it
misses the floor -- but it is the project's benchmark everywhere else and
dropping it silently would be the wrong call. Putting it in the main set
instead would force the shared round count from 8.1k down to 2.9k and leave
nothing significant anywhere, which is why it is an aside and not a row."""


def ogd_weights(P_s, aw_s, y, freeze_at=None):
    """run_ogd_sleeping's recursion, with an optional freeze point. After
    `freeze_at` the weights are still projected onto each round's awake subset
    (needed to predict) but never stepped (no learning)."""
    D, N = np.sqrt(2.0), P_s.shape[1]
    loss = per_round_expert_loss(P_s, y)
    w = np.full(N, 1.0 / N)
    W = np.zeros((len(y), N))
    for t in range(len(y)):
        a = aw_s[t]
        if not a.any():
            continue
        v = project_to_simplex(w[a])
        W[t, a] = v
        if freeze_at is not None and t >= freeze_at:
            continue
        w[a] = v
        w[a] = w[a] - (OGD_STEP_MULTIPLIER * D / (M * np.sqrt(t + 1))) * loss[t, a]
    return W


def calibrated_loss(probs, y, mask, train, T, live):
    """Per-round calibrated log-loss at full length. The calibration rate is
    chosen on TRAIN rounds only; `live` decides whether the temperature keeps
    adapting after the cut or is frozen at its value there."""
    p, yy = probs[mask], y[mask]
    n_tr = int(train[mask].sum())
    c, _ = select_lr_on_train(p[:n_tr], yy[:n_tr], LR_GRID)
    if live:
        cal, _ = temperature_calibrate(p, yy, lr_scale=c)
    else:
        cal_tr, s_hist = temperature_calibrate(p[:n_tr], yy[:n_tr], lr_scale=c)
        e = np.exp(s_hist[-1] * np.log(np.clip(p, EPS, 1.0)))
        cal = e / e.sum(axis=1, keepdims=True)
        cal[:n_tr] = cal_tr
    out = np.full(T, np.nan)
    out[mask] = -np.log(np.clip(cal[np.arange(len(yy)), yy], EPS, 1.0))
    return out


def plot_deployment(table, out_path, n_test):
    df = table.sort_values("log_loss").reset_index(drop=True)
    colors = {"αλγόριθμος": RANK_COLORS["algorithm"], "closing": RANK_COLORS["bookmaker"],
              "opening": "#c3c2b7"}
    fig, ax = plt.subplots(figsize=(9, 0.46 * len(df) + 2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ypos = np.arange(len(df))[::-1]
    for yp, (_, r) in zip(ypos, df.iterrows()):
        ax.scatter([r["log_loss"]], [yp], s=80, color=colors[r["type"]],
                   edgecolor="#fcfcfb", linewidth=1.1, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(df["name"], fontsize=9, color="#0b0b0b")
    ax.set_xlabel(f"Βαθμονομημένο log-loss, {n_test} αγώνες του 2025+", color="#52514e")
    ax.set_title("Εκπαίδευση ως 31/12/2024, πρόβλεψη 2025--2026",
                 color="#0b0b0b", fontsize=12, pad=12)
    _style_axes(ax)
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def main():
    panel, P, awake, y, dates, seasons = load_universe_from(
        os.path.join(DATA_DIR, "odds_long.csv"))
    T, N, _ = P.shape
    test = dates >= CUT
    train = ~test
    cut_idx = int(np.flatnonzero(test)[0])
    print(f"train: {int(train.sum())} γύροι ως {str(dates[train].max())[:10]}")
    print(f"test : {int(test.sum())} γύροι, {str(dates[test].min())[:10]} -> "
          f"{str(dates[test].max())[:10]}\n")

    _, close_idx = split_by_market_phase(panel)
    series = {}
    for pname, idx in (("πλήρες panel", list(range(N))), ("closing-only", close_idx)):
        P_s, aw_s = P[:, idx, :], awake[:, idx]
        alive = aw_s.any(axis=1)
        for mode, fr, live in (("online", None, True), ("παγωμένο", cut_idx, False)):
            W = ogd_weights(P_s, aw_s, y, freeze_at=fr)
            phat = np.einsum("tn,tnk->tk", W, P_s)
            series[f"OGD {pname}, {mode}"] = {
                "loss": calibrated_loss(phat, y, alive, train, T, live),
                "type": "αλγόριθμος"}

    cov_test = awake[test].mean(axis=0) * 100
    cov, books = {}, []
    for k, b in enumerate(panel):
        multi = len(set(seasons[awake[:, k]])) > 1
        if multi and (cov_test[k] >= COVERAGE_FLOOR or b in ASIDE):
            series[b] = {"loss": calibrated_loss(P[:, k, :], y, awake[:, k], train, T, True),
                         "type": "closing" if b.endswith("C") else "opening"}
            cov[b] = cov_test[k]
            if b not in ASIDE:
                books.append(b)
    print("αντίπαλοι στο test: " + ", ".join(f"{b} {cov[b]:.0f}%" for b in books))
    for b in ASIDE:
        print(f"  ξεχωριστά -- {b}: κάλυψη {cov[b]:.1f}%, τελευταία απόδοση "
              f"{str(dates[awake[:, panel.index(b)]].max())[:10]}")

    algos = [k for k, s in series.items() if s["type"] == "αλγόριθμος"]
    defined = np.ones(T, bool)
    for k in algos + books:
        defined &= ~np.isnan(series[k]["loss"])
    sel = test & defined
    n_test = int(sel.sum())
    print(f"\nκοινοί γύροι: {n_test}\n")

    pd.set_option("display.width", 200)
    table = pd.DataFrame([
        {"name": k, "type": series[k]["type"], "coverage_test_pct": cov.get(k, 100.0),
         "log_loss": float(np.nanmean(np.where(sel, series[k]["loss"], np.nan)))}
        for k in algos + books]).sort_values("log_loss").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    rows = []
    for name in algos:
        s = series[name]
        for b in books + list(ASIDE):
            aside = b in ASIDE
            # the main set shares one round set; Pinnacle is judged one-to-one
            # on its own overlap, which is the only way it stays in at all
            m = (test & ~np.isnan(s["loss"]) & ~np.isnan(series[b]["loss"])) if aside else sel
            d, n = paired_diff(np.where(m, s["loss"], np.nan),
                               np.where(m, series[b]["loss"], np.nan))
            r = moving_block_bootstrap_test(d)
            rows.append({"variant": name, "bookmaker": b, "phase": series[b]["type"],
                         "aside": aside, "n": n,
                         "verdict": ("ισοπαλία" if not r["significant_95"]
                                     else ("αλγόριθμος" if r["mean_diff"] < 0 else b)),
                         **r})
    sig = pd.DataFrame(rows)
    cols = ["variant", "bookmaker", "phase", "n", "mean_diff", "ci_low", "ci_high",
            "p_value", "verdict"]
    print("\n--- ζευγαρωτά, κύριο σύνολο (αρνητικό = ο αλγόριθμος καλύτερος) ---")
    print(sig[~sig["aside"]][cols].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nσύνοψη κύριου συνόλου:")
    for v in sig["variant"].unique():
        s = sig[(sig["variant"] == v) & (~sig["aside"])]
        w = int((s["verdict"] == "αλγόριθμος").sum())
        l = int((~s["verdict"].isin(["αλγόριθμος", "ισοπαλία"])).sum())
        print(f"  {v:34s}: {w} νίκες, {l} ήττες, {len(s) - w - l} ισοπαλίες")

    print(f"\n--- ΣΗΜΕΙΩΣΗ: {', '.join(ASIDE)} ένα προς ένα, στους δικούς τους γύρους ---")
    print(sig[sig["aside"]][cols].to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    tp = os.path.join(OUT_DIR, "deployment_test.csv")
    sp = os.path.join(OUT_DIR, "deployment_test_significance.csv")
    table.to_csv(tp, index=False)
    sig.to_csv(sp, index=False)
    print(f"\nSaved: {tp}\nSaved: {sp}")
    plot_deployment(table, os.path.join(OUT_DIR, "deployment_test.png"), n_test)


if __name__ == "__main__":
    main()
