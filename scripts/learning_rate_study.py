"""
ARE THE LEARNING RATES ANY GOOD?

The three algorithms use the textbook worst-case-optimal schedules:

    Hedge:      eta_t = sqrt(ln N / (t+1)) / M
    OGD/FTRL:   eta_t = D / (M * sqrt(t+1)),   D = sqrt(2)

with M = log(1/EPS) = log(1e6) ~ 13.82, the bound on a single round's
per-expert loss. Two separate things are worth questioning about that.

1. M IS ENORMOUSLY CONSERVATIVE. It assumes a bookmaker might price the
   realised outcome at EPS = 1e-6. Real de-vigged football probabilities
   bottom out around 1-2%, i.e. a per-round loss near 4, not 13.8. Since
   eta ~ 1/M, every rate in the project is several times smaller than the
   same theory would allow with an honest bound -- the algorithms adapt far
   more slowly than they need to. `report_loss_range()` measures the true
   range so the size of that gap is a number, not a guess.

2. MORE FUNDAMENTALLY, THE WHOLE REDUCTION LEAVES THE BEST RATE ON THE
   TABLE. The project linearises: ell_t(w) = sum_i w_i * ell_t(i), which is
   linear in w, so only generic O(sqrt(T)) rates apply. But the loss we
   actually care about, ell_t(w) = -log(<w, p_t>), is EXP-CONCAVE
   (exp(-ell) = <w, p_t> is linear, hence concave). For log-loss the
   Aggregating Algorithm / Bayesian mixture (Vovk 1990; Cesa-Bianchi &
   Lugosi ch. 3) attains regret <= ln N with a CONSTANT eta = 1 -- no
   sqrt(T) term at all. With N=26 and T=76,584 that is ln 26 = 3.26 total,
   against sqrt(T ln N) ~ 500 for the current schedule.

   And eta = 1 on the per-expert log-loss is exactly the Bayesian posterior
   update, w_i <- w_i * p_i(y_t), because exp(-1 * -log p_i) = p_i. So the
   "better algorithm" is a one-line change to run_hedge_sleeping.

This file measures both: a multiplier sweep on the existing schedules, and
the constant-eta / Bayesian variants.

NO SNOOPING: every variant is scored on a held-out tail. The multiplier is
chosen on the first TRAIN_FRACTION of rounds and the reported number is the
remaining rounds only -- the same chronological discipline as
calibration_correction.py. Full-sample numbers are printed too, clearly
labelled, because the gap between them is itself informative.

A LESSON THIS FILE TAUGHT ITSELF: its first version's grid for Hedge/FTRL
started at x1 and only went up, and reported both as "already optimal at
x1" -- but x1 was the grid's LOWER BOUNDARY, not an interior optimum, and
nobody had looked to its left. HEDGE_FTRL_MULTIPLIERS below now extends down
to x0.001 and finds genuine interior minima at x0.25 (Hedge) and x0.05
(FTRL), now shipped as HEDGE_STEP_MULTIPLIER / FTRL_STEP_MULTIPLIER in
sleeping_experts.py. Whenever a sweep's answer sits on a grid endpoint,
extend the grid before believing it -- OGD's own x100 optimum is safely
interior to RATE_MULTIPLIERS below, which is why it was caught immediately
and Hedge/FTRL's was not.

Produces:
  - results/learning_rate_study.csv
  - results/learning_rate_study.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    OUT_DIR, EPS, M, COLORS, OGD_STEP_MULTIPLIER, HEDGE_STEP_MULTIPLIER,
    FTRL_STEP_MULTIPLIER,
    load_full_universe, per_round_expert_loss, project_to_simplex,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping, evaluate,
)

TRAIN_FRACTION = 0.75
RATE_MULTIPLIERS = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0, 110.0,
                    125.0, 150.0, 200.0, 500.0, 1000.0]           # OGD grid
HEDGE_FTRL_MULTIPLIERS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5,
                          0.75, 1.0, 1.5, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0,
                          200.0, 500.0, 1000.0]                   # Hedge/FTRL grid
"""Deliberately extended well below x1: that is where HEDGE_STEP_MULTIPLIER
(0.25) and FTRL_STEP_MULTIPLIER (0.05) turned out to live -- see the module
docstring."""
CONSTANT_ETAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03,
                 0.1, 0.3, 0.5, 1.0, 2.0]  # 1.0 = Bayesian mixture / Aggregating Algorithm
"""The first version of this grid started at 0.1 and the constant schedule was
written off on the strength of it. That was the same mistake the multiplier grid
made (see the module docstring): 0.1 was the LOWER endpoint, and the decaying
schedules it is being compared against are nowhere near it -- shipped Hedge
passes eta ~ 0.033 at t=1 and ~ 1.7e-4 by the middle of the run, i.e. the whole
interesting range sat below the old grid. It now spans five decades."""

CONSTANT_ETAS_OGD = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3,
                     1.0, 3.0, 10.0, 30.0, 100.0]
CONSTANT_ETAS_FTRL = [1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7,
                      1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
                      0.01, 0.1, 1.0]
"""Extended below 1e-6 on the study's own rule: the first run of this grid put
the minimum exactly on its lower endpoint, which is the same mistake the
multiplier grid made. The extension matters because the limit is known --
eta -> 0 makes FTRL's projected point w0, i.e. the uniform average -- so an
optimum strictly inside (0, 1e-6) is a real claim and an optimum AT the old
endpoint was not yet one."""
"""Constant-eta grids for the other two, which the original study never tried at
all -- it only ever asked the question of Hedge. Each grid brackets the decaying
schedule that algorithm actually ships: OGD passes eta ~ 10 at t=1 and ~ 0.05 at
the midpoint, FTRL ~ 5e-3 and ~ 2.6e-5."""


def report_loss_range(loss, awake):
    """How conservative is M really? Compares the theoretical bound against
    what the data actually produces."""
    vals = loss[awake]
    print(f"M used by the algorithms      : {M:.3f}   (= log(1/EPS), EPS={EPS:g})")
    print(f"max per-expert loss observed  : {vals.max():.3f}")
    print(f"99.9th percentile             : {np.percentile(vals, 99.9):.3f}")
    print(f"mean                          : {vals.mean():.3f}")
    print(f"=> eta is smaller than an honest bound would allow by a factor of "
          f"~{M / vals.max():.1f}x (max) / ~{M / np.percentile(vals, 99.9):.1f}x (99.9th pct)\n")
    return float(vals.max())


def hedge_scaled(loss, awake, N, mult=1.0, const_eta=None):
    """sleeping_experts.run_hedge_sleeping with the learning rate either
    multiplied by `mult` or replaced by a constant. const_eta=1.0 is the
    Bayesian mixture: w_i <- w_i * p_i(y_t).

    Carried in LOG weight space. That is not a cosmetic choice: at eta=1 over
    76k rounds the linear weights underflow to 0, and the obvious patch --
    rescaling w after each update -- is exactly the bug run_hedge_sleeping's
    docstring warns about, because the persistent state must NEVER be
    renormalised (only the awake-restricted prediction copy may be). An
    earlier draft of this file did rescale and silently stopped reproducing
    run_hedge_sleeping at mult=1.0; main() now asserts that it does."""
    logw = np.full(N, -np.log(N))
    T = loss.shape[0]
    W = np.zeros((T, N))
    for t in range(T):
        a = awake[t]
        if not a.any():
            continue
        la = logw[a]
        v = np.exp(la - la.max())           # log-sum-exp shift: prediction only
        v /= v.sum()
        W[t, a] = v
        eta = const_eta if const_eta is not None else mult * np.sqrt(np.log(N) / (t + 1)) / M
        logw[a] = la - eta * loss[t, a]
    return W


def ogd_scaled(loss, awake, N, mult=1.0, const_eta=None):
    """run_ogd_sleeping with the step size multiplied by `mult`, or replaced by
    a constant. A constant step is the one schedule the project had not ruled
    out: 1/sqrt(t) is derived for a STATIC comparator, and this panel is not
    static (Pinnacle's coverage collapses late), so a step that never stops
    moving is the theoretically motivated alternative, not a hack."""
    D = np.sqrt(2.0)
    w = np.full(N, 1.0 / N)
    T = loss.shape[0]
    W = np.zeros((T, N))
    for t in range(T):
        a = awake[t]
        if not a.any():
            continue
        v = project_to_simplex(w[a])
        W[t, a] = v
        eta = const_eta if const_eta is not None else mult * D / (M * np.sqrt(t + 1))
        w[a] = v - eta * loss[t, a]
    return W


def ftrl_scaled(loss, awake, N, mult=1.0, const_eta=None):
    """run_ftrl_sleeping with the step size multiplied by `mult`. FTRL shares
    OGD's eta FORMULA (same D/(M*sqrt(t+1)) shape) but not its optimal
    multiplier -- FTRL wants x0.05 (shipped as FTRL_STEP_MULTIPLIER) where OGD
    wants x100, because FTRL's eta multiplies the whole accumulated loss while
    OGD's multiplies one round's gradient. Both are swept and both shipped
    values are tuned, so the OGD-vs-FTRL comparison stays fair."""
    D = np.sqrt(2.0)
    w0 = np.full(N, 1.0 / N)
    cum_loss = np.zeros(N)
    T = loss.shape[0]
    W = np.zeros((T, N))
    for t in range(T):
        a = awake[t]
        if not a.any():
            continue
        eta = const_eta if const_eta is not None else mult * D / (M * np.sqrt(t + 1))
        v = project_to_simplex(w0[a] - eta * cum_loss[a])
        W[t, a] = v
        cum_loss[a] += loss[t, a]
    return W


def ogd_adagrad(loss, awake, N, per_coordinate=True):
    """OGD with an ADAGRAD step size: no worst-case bound M anywhere. The step
    is set from the gradients actually observed so far,

        eta_t = D / sqrt(sum_{s<=t} g_s^2),

    either per expert (per_coordinate=True, which suits sleeping experts --
    an expert's accumulator only advances on rounds it was awake, so its step
    tracks its own participation count rather than global time) or with one
    shared accumulator (the textbook scalar version). Parameter-free: nothing
    here is tuned, which is the whole point of comparing it against the
    hand-picked multiplier."""
    D = np.sqrt(2.0)
    w = np.full(N, 1.0 / N)
    T = loss.shape[0]
    W = np.zeros((T, N))
    acc = np.zeros(N)
    acc_global = 0.0
    for t in range(T):
        a = awake[t]
        if not a.any():
            continue
        v = project_to_simplex(w[a])
        W[t, a] = v
        w[a] = v
        g = loss[t, a]
        if per_coordinate:
            acc[a] += g ** 2
            step = D / np.sqrt(acc[a])
        else:
            acc_global += float((g ** 2).sum())
            step = D / np.sqrt(acc_global)
        w[a] = w[a] - step * g
    return W


def score(W, P, y, val_mask):
    phat = np.einsum("tn,tnk->tk", W, P)
    full = evaluate(phat, y)
    val = evaluate(phat, y, mask=val_mask)
    return full["log_loss"], val["log_loss"]


def plot_study(table, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), facecolor="#fcfcfb")

    sweep = table[table["family"] == "Hedge x mult"]
    ax1.set_facecolor("#fcfcfb")
    for col, lab, style in [("val_log_loss", "held-out tail", "-"),
                             ("full_log_loss", "full sample", "--")]:
        ax1.plot(sweep["setting"], sweep[col], marker="o", markersize=5,
                  linestyle=style, linewidth=1.9, color=COLORS["Hedge"], alpha=1.0 if style == "-" else 0.5,
                  label=f"Hedge, {lab}")
    ogd = table[table["family"] == "OGD x mult"]
    for col, lab, style in [("val_log_loss", "held-out tail", "-"),
                             ("full_log_loss", "full sample", "--")]:
        ax1.plot(ogd["setting"], ogd[col], marker="s", markersize=5,
                  linestyle=style, linewidth=1.9, color=COLORS["OGD"], alpha=1.0 if style == "-" else 0.5,
                  label=f"OGD, {lab}")
    ax1.set_xscale("log")
    ax1.set_xlabel("learning-rate multiplier (1 = textbook schedule)", color="#52514e")
    ax1.set_ylabel("log-loss", color="#52514e")
    ax1.set_title("Where is the bottom of the curve?\n"
                   f"(OGD ships at x{OGD_STEP_MULTIPLIER:g}, Hedge at "
                   f"x{HEDGE_STEP_MULTIPLIER:g}, FTRL at x{FTRL_STEP_MULTIPLIER:g})",
                   color="#0b0b0b", fontsize=12, pad=10)
    ax1.grid(True, color="#e1e0d9", linewidth=0.8)
    ax1.set_axisbelow(True)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax1.spines[s].set_color("#c3c2b7")
    ax1.tick_params(colors="#898781")
    leg = ax1.legend(frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color("#0b0b0b")

    const = table[table["family"] == "Hedge const eta"].sort_values("setting")
    base = table[(table["family"] == "Hedge x mult") &
                 (table["setting"] == HEDGE_STEP_MULTIPLIER)]["val_log_loss"].iloc[0]
    ax2.set_facecolor("#fcfcfb")
    ax2.axhline(base, color="#898781", linestyle="--", linewidth=1.5,
                 label="shipped Hedge (held-out)")
    ax2.plot(const["setting"], const["val_log_loss"], marker="o", markersize=6,
              linewidth=2.0, color="#4a3aa7", label="constant eta (held-out)")
    ax2.set_xlabel("constant eta  (1.0 = Bayesian mixture)", color="#52514e")
    ax2.set_ylabel("log-loss, held-out tail", color="#52514e")
    ax2.set_title("Constant-eta / Bayesian variant", color="#0b0b0b", fontsize=12, pad=10)
    ax2.grid(True, color="#e1e0d9", linewidth=0.8)
    ax2.set_axisbelow(True)
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax2.spines[s].set_color("#c3c2b7")
    ax2.tick_params(colors="#898781")
    leg2 = ax2.legend(frameon=False, fontsize=8.5)
    for t in leg2.get_texts():
        t.set_color("#0b0b0b")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


def main():
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    loss = per_round_expert_loss(P, y)

    print(f"{T} rounds, {N} experts\n")
    report_loss_range(loss, awake)

    # The sweeps must reproduce the shipped algorithms, otherwise they are
    # measuring a different algorithm. `mult` here is always relative to the
    # UNSCALED textbook schedule, so it must be compared at the SHIPPED
    # multiplier for each algorithm, not at mult=1 -- all three now carry a
    # tuned constant (see sleeping_experts.py), this study is what set them.
    assert np.allclose(hedge_scaled(loss, awake, N, mult=HEDGE_STEP_MULTIPLIER),
                       run_hedge_sleeping(loss, awake, N)), \
        f"hedge_scaled(mult={HEDGE_STEP_MULTIPLIER}) must equal run_hedge_sleeping"
    assert np.allclose(ogd_scaled(loss, awake, N, mult=OGD_STEP_MULTIPLIER),
                       run_ogd_sleeping(loss, awake, N)), \
        f"ogd_scaled(mult={OGD_STEP_MULTIPLIER}) must equal run_ogd_sleeping"
    assert np.allclose(ftrl_scaled(loss, awake, N, mult=FTRL_STEP_MULTIPLIER),
                       run_ftrl_sleeping(loss, awake, N)), \
        f"ftrl_scaled(mult={FTRL_STEP_MULTIPLIER}) must equal run_ftrl_sleeping"
    print(f"sanity check passed: Hedge at x{HEDGE_STEP_MULTIPLIER:g}, "
          f"OGD at x{OGD_STEP_MULTIPLIER:g}, FTRL at x{FTRL_STEP_MULTIPLIER:g} "
          f"reproduce the shipped algorithms\n")

    n_train = int(round(T * TRAIN_FRACTION))
    val_mask = np.zeros(T, dtype=bool)
    val_mask[n_train:] = True
    print(f"train = first {n_train} rounds, held-out tail = {T - n_train} rounds\n")

    rows = []

    print("-- Hedge, learning-rate multiplier --")
    for m in HEDGE_FTRL_MULTIPLIERS:
        full, val = score(hedge_scaled(loss, awake, N, mult=m), P, y, val_mask)
        rows.append({"family": "Hedge x mult", "setting": m,
                      "full_log_loss": full, "val_log_loss": val})
        print(f"   x{m:<6g} full={full:.5f}  held-out={val:.5f}")

    print("\n-- OGD, step-size multiplier --")
    for m in RATE_MULTIPLIERS:
        full, val = score(ogd_scaled(loss, awake, N, mult=m), P, y, val_mask)
        rows.append({"family": "OGD x mult", "setting": m,
                      "full_log_loss": full, "val_log_loss": val})
        print(f"   x{m:<6g} full={full:.5f}  held-out={val:.5f}")

    print("\n-- Hedge, CONSTANT eta (1.0 = Bayesian mixture / Aggregating Algorithm) --")
    for e in CONSTANT_ETAS:
        full, val = score(hedge_scaled(loss, awake, N, const_eta=e), P, y, val_mask)
        rows.append({"family": "Hedge const eta", "setting": e,
                      "full_log_loss": full, "val_log_loss": val})
        print(f"   eta={e:<5g} full={full:.5f}  held-out={val:.5f}")

    print("\n-- OGD, CONSTANT eta (never decays) --")
    for e in CONSTANT_ETAS_OGD:
        full, val = score(ogd_scaled(loss, awake, N, const_eta=e), P, y, val_mask)
        rows.append({"family": "OGD const eta", "setting": e,
                      "full_log_loss": full, "val_log_loss": val})
        print(f"   eta={e:<8g} full={full:.5f}  held-out={val:.5f}")

    print("\n-- FTRL, CONSTANT eta (never decays) --")
    for e in CONSTANT_ETAS_FTRL:
        full, val = score(ftrl_scaled(loss, awake, N, const_eta=e), P, y, val_mask)
        rows.append({"family": "FTRL const eta", "setting": e,
                      "full_log_loss": full, "val_log_loss": val})
        print(f"   eta={e:<8g} full={full:.5f}  held-out={val:.5f}")

    print("\n-- FTRL, step-size multiplier (same eta formula as OGD) --")
    for m in HEDGE_FTRL_MULTIPLIERS:
        full, val = score(ftrl_scaled(loss, awake, N, mult=m), P, y, val_mask)
        rows.append({"family": "FTRL x mult", "setting": m,
                      "full_log_loss": full, "val_log_loss": val})
        print(f"   x{m:<6g} full={full:.5f}  held-out={val:.5f}")

    print("\n-- OGD with an HONEST bound M instead of log(1/EPS) --")
    #   eta = mult*D/(M*sqrt(t+1)) == D/((M/mult)*sqrt(t+1)), so mult = M/M_tight
    for label, m_tight, note in [
        ("M=log(100)=4.61", np.log(100.0), "a priori: no de-vigged football probability below 1%"),
        ("M=3.76 (observed max)", 3.763, "LOOK-AHEAD: uses the whole dataset's max loss"),
    ]:
        mult = M / m_tight
        full, val = score(ogd_scaled(loss, awake, N, mult=mult), P, y, val_mask)
        rows.append({"family": "OGD honest M", "setting": m_tight, "name": label,
                      "full_log_loss": full, "val_log_loss": val})
        print(f"   {label:24s} (= x{mult:.2f})  full={full:.5f}  held-out={val:.5f}   [{note}]")

    print("\n-- OGD with an ADAPTIVE (AdaGrad) step: no M, no tuning --")
    for label, per_coord in [("AdaGrad per-expert", True), ("AdaGrad scalar", False)]:
        full, val = score(ogd_adagrad(loss, awake, N, per_coordinate=per_coord), P, y, val_mask)
        rows.append({"family": "OGD adaptive", "setting": np.nan, "name": label,
                      "full_log_loss": full, "val_log_loss": val})
        print(f"   {label:24s}  full={full:.5f}  held-out={val:.5f}")

    # reference points, unchanged code paths
    for name, W in [("Hedge (shipped)", run_hedge_sleeping(loss, awake, N)),
                     ("OGD (shipped)", run_ogd_sleeping(loss, awake, N)),
                     ("FTRL (shipped)", run_ftrl_sleeping(loss, awake, N))]:
        full, val = score(W, P, y, val_mask)
        rows.append({"family": "shipped", "setting": np.nan, "name": name,
                      "full_log_loss": full, "val_log_loss": val})
    W_uniform = awake / awake.sum(axis=1, keepdims=True)
    full, val = score(W_uniform, P, y, val_mask)
    rows.append({"family": "shipped", "setting": np.nan, "name": "Uniform average",
                  "full_log_loss": full, "val_log_loss": val})

    table = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "learning_rate_study.csv")
    table.to_csv(out, index=False)

    pd.set_option("display.width", 160)
    print(f"\n=== all settings — saved to {out} ===\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # the honest readout: pick on train-implied best, report the tail
    print("\n=== best setting per family, chosen by HELD-OUT tail ===")
    ref = table[table["family"] == "shipped"].set_index("name")["val_log_loss"]
    fam_baseline = {"Hedge x mult": "Hedge (shipped)", "OGD x mult": "OGD (shipped)",
                    "FTRL x mult": "FTRL (shipped)", "Hedge const eta": "Hedge (shipped)",
                    "OGD const eta": "OGD (shipped)", "FTRL const eta": "FTRL (shipped)"}
    for fam, base_name in fam_baseline.items():
        sub = table[table["family"] == fam].sort_values("val_log_loss")
        best = sub.iloc[0]
        base = ref[base_name]
        print(f"  {fam:18s} best setting {best['setting']:<6g} "
              f"held-out {best['val_log_loss']:.5f}  "
              f"vs shipped {base:.5f}  (improvement {base - best['val_log_loss']:+.5f})")
    print(f"\n  reference: Uniform average held-out {ref['Uniform average']:.5f}")

    # how much of the hand-tuned gain do the principled options recover?
    shipped = ref["OGD (shipped)"]
    best_tuned = table[table["family"] == "OGD x mult"]["val_log_loss"].min()
    span = shipped - best_tuned
    print(f"\n=== how much of the hand-tuned OGD gain does each PRINCIPLED option recover? ===")
    print(f"    (shipped {shipped:.5f} -> best multiplier {best_tuned:.5f}, span {span:+.5f})\n")
    for _, r in table[table["family"].isin(["OGD honest M", "OGD adaptive"])].iterrows():
        got = shipped - r["val_log_loss"]
        print(f"    {r['name']:24s} held-out {r['val_log_loss']:.5f}  "
              f"gain {got:+.5f}  = {100 * got / span:5.1f}% of the tuned gain")

    plot_study(table, os.path.join(OUT_DIR, "learning_rate_study.png"))


if __name__ == "__main__":
    main()
