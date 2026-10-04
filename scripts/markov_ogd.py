"""
MARKOV CHAIN ONLINE GRADIENT DESCENT.

A generalisation of OGD in which, after the usual gradient step and projection,
the weight mass is moved by a row-stochastic transition matrix M:

    v_t      = Pi_simplex(w_t)                 # prediction weights
    w_hat    = v_t - eta_t * loss_t            # gradient step
    w_{t+1}  = M^T w_hat                       # Markov transition

Fixed-Share is the special case M = (1-a) I + (a/k) 11^T -- a uniform leak.
`no_pinnacle_panel.py` and the joint alpha/step sweep both found that uniform
leak buys nothing once the step itself is tuned, because both knobs control the
same quantity: how fast the algorithm forgets. A transition matrix is strictly
more expressive, because it can say WHERE the forgotten mass should go.

That matters here because the panel has structure the uniform leak throws away:

  * the 24 experts are 12 PAIRS of the same firm at two points in time
    (B365 / B365C), so a firm's opening and closing quote are far from
    independent -- mass leaving one has an obvious place to go;
  * the panel splits into two PHASES whose members behave alike within a phase
    and systematically differently across phases (see the opening/closing
    results).

Four kernels are compared, all restricted to the awake subset each round (the
Sleeping Experts reduction freezes everyone else, so the chain must too):

  identity  M = I                       -- must reproduce plain OGD exactly
  uniform   leak toward all awake       -- the Fixed-Share analogue
  pair      leak toward one's own counterpart (B365 <-> B365C)
  phase     leak toward others of the same market phase

SELECTION: the mixing rate is chosen on the chronological train prefix only,
the same protocol as every other hyperparameter in the project. Held-out
numbers are reported but never used to choose.

Produces:
  - results/markov_ogd.csv
  - results/markov_ogd_significance.csv
  - results/markov_ogd.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sleeping_experts import (
    OUT_DIR, EPS, M as LOSS_BOUND, COLORS,
    load_full_universe, per_round_expert_loss, project_to_simplex,
    run_hedge_sleeping, run_ogd_sleeping, run_ftrl_sleeping, evaluate,
    OGD_STEP_MULTIPLIER,
)
from calibration_analysis import _style_axes
from final_ranking import calibrate_full_history
from significance_test import moving_block_bootstrap_test, paired_diff

TRAIN_FRACTION = 0.75
ALPHAS = (0.0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)
KERNELS = ("identity", "uniform", "pair", "phase")


# ------------------------------------------------------------ the kernels --

def build_structure(panel):
    """Per-expert counterpart index and market phase.

    A name ending in C is the closing quote of the identically-prefixed opening
    one -- football-data.co.uk's own convention, preserved by data_processer.py.
    """
    idx = {n: k for k, n in enumerate(panel)}
    counterpart = np.full(len(panel), -1, dtype=int)
    is_closing = np.zeros(len(panel), dtype=bool)
    for k, n in enumerate(panel):
        is_closing[k] = n.endswith("C")
        other = n[:-1] if n.endswith("C") else n + "C"
        if other in idx:
            counterpart[k] = idx[other]
    return counterpart, is_closing


def transition(kind, alpha, awake_idx, counterpart, is_closing):
    """Row-stochastic matrix over the AWAKE subset only.

    Rows index the expert losing mass, columns the one receiving it. Every row
    sums to 1, so total mass is preserved exactly -- which is what makes
    alpha=0 a true no-op rather than an approximate one.
    """
    k = len(awake_idx)
    Mm = np.eye(k)
    if kind == "identity" or alpha == 0.0 or k == 1:
        return Mm
    pos = {g: i for i, g in enumerate(awake_idx)}

    if kind == "uniform":
        Mm = (1.0 - alpha) * np.eye(k) + alpha / k
        return Mm

    if kind == "pair":
        Mm = (1.0 - alpha) * np.eye(k)
        for i, g in enumerate(awake_idx):
            c = counterpart[g]
            if c >= 0 and c in pos:
                Mm[i, pos[c]] += alpha          # counterpart is awake: send it there
            else:
                Mm[i, i] += alpha               # otherwise keep it
        return Mm

    if kind == "phase":
        Mm = (1.0 - alpha) * np.eye(k)
        for phase in (False, True):
            same = [i for i, g in enumerate(awake_idx) if is_closing[g] == phase]
            if not same:
                continue
            share = alpha / len(same)
            for i in same:
                for j in same:
                    Mm[i, j] += share
        # experts alone in their phase keep their own mass
        for i in range(k):
            s = Mm[i].sum()
            if not np.isclose(s, 1.0):
                Mm[i, i] += 1.0 - s
        return Mm

    raise ValueError(kind)


# ------------------------------------------------------------- algorithm --

def run_markov_ogd(loss, awake, N, kind, alpha, counterpart, is_closing):
    """OGD with a Markov transition, under the Sleeping Experts reduction."""
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
        w[a] = v
        eta = OGD_STEP_MULTIPLIER * D / (LOSS_BOUND * np.sqrt(t + 1))
        w[a] = w[a] - eta * loss[t, a]
        if kind != "identity" and alpha > 0.0:
            ai = np.flatnonzero(a)
            Mm = transition(kind, alpha, ai, counterpart, is_closing)
            w[a] = Mm.T @ w[a]
    return W


# ------------------------------------------------------------------ main --

def effective_sample_schedule(loss, awake, N, mult, tau):
    """OGD whose step decays on the EFFECTIVE sample count t/tau rather than on
    t itself.

    This is the actionable consequence of Markov Chain Gradient Descent in the
    sense of Sun, Sun & Yin (2018): the samples here are not i.i.d. draws but
    successive points on a dependent trajectory, so by round t the algorithm has
    seen fewer than t independent observations. If the dependence horizon is
    tau rounds, the effective count is ~t/tau and the schedule becomes

        eta_t  =  mult * D / (M * sqrt(t/tau + 1))  =  ~sqrt(tau) * eta_t^{iid}

    i.e. a dependent stream justifies a step inflated by roughly sqrt(tau).
    That is a *theoretical* route to the same kind of correction that
    learning_rate_study.py found empirically, and the point of running it is to
    see how much of the empirical factor it accounts for.
    """
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
        w[a] = v
        eta = mult * D / (LOSS_BOUND * np.sqrt(t / tau + 1.0))
        w[a] = w[a] - eta * loss[t, a]
    return W


def main():
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    loss = per_round_expert_loss(P, y)
    cut = int(round(T * TRAIN_FRACTION))
    counterpart, is_closing = build_structure(panel)
    n_pairs = int((counterpart >= 0).sum() // 2)
    print(f"{T} γύροι, {N} experts, {n_pairs} ζεύγη opening/closing")
    print(f"train ως τον γύρο {cut}, ουρά {T - cut}\n")

    def split_scores(W):
        phat = np.einsum("tn,tnk->tk", W, P)
        ll = -np.log(np.clip(phat[np.arange(T), y], EPS, 1.0))
        return ll[:cut].mean(), ll[cut:].mean(), phat

    # sanity: the identity kernel must BE plain OGD
    W_id = run_markov_ogd(loss, awake, N, "identity", 0.0, counterpart, is_closing)
    assert np.allclose(W_id, run_ogd_sleeping(loss, awake, N)), \
        "identity kernel must reproduce run_ogd_sleeping exactly"
    print("sanity ok: identity kernel == run_ogd_sleeping\n")

    rows = []
    for kind in KERNELS:
        alphas = (0.0,) if kind == "identity" else ALPHAS
        for a in alphas:
            W = run_markov_ogd(loss, awake, N, kind, a, counterpart, is_closing)
            tr, va, _ = split_scores(W)
            rows.append({"kernel": kind, "alpha": a, "train": tr, "held_out": va})
            print(f"  {kind:9s} α={a:<7g} train {tr:.6f}  ουρά {va:.6f}")
    sweep = pd.DataFrame(rows)

    print("\n=== επιλογή στο train ανά πυρήνα ===")
    best = {}
    for kind in KERNELS:
        s = sweep[sweep["kernel"] == kind]
        b = s.loc[s["train"].idxmin()]
        best[kind] = float(b["alpha"])
        print(f"  {kind:9s} α={b['alpha']:<7g} -> ουρά {b['held_out']:.6f}")

    # ---- full comparison against the shipped algorithms, same rounds ----
    entries = {}
    for kind in KERNELS:
        label = "OGD" if kind == "identity" else f"MC-OGD ({kind}, α={best[kind]:g})"
        entries[label] = run_markov_ogd(loss, awake, N, kind, best[kind],
                                        counterpart, is_closing)
    entries["Hedge"] = run_hedge_sleeping(loss, awake, N)
    entries["FTRL"] = run_ftrl_sleeping(loss, awake, N)
    entries["Uniform average"] = np.divide(
        awake, awake.sum(axis=1, keepdims=True),
        out=np.zeros_like(awake, dtype=float),
        where=awake.sum(axis=1, keepdims=True) > 0)

    alive = np.ones(T, dtype=bool)
    table, per_round = [], {}
    for label, W in entries.items():
        phat = np.einsum("tn,tnk->tk", W, P)
        p, cal, yy, _ = calibrate_full_history(phat, y, alive)
        raw_m, cal_m = evaluate(p, yy), evaluate(cal, yy)
        per_round[label] = -np.log(np.clip(cal[np.arange(T), yy], EPS, 1.0))
        table.append({
            "name": label,
            "raw_log_loss": raw_m["log_loss"], "calibrated_log_loss": cal_m["log_loss"],
            "raw_brier": raw_m["brier"], "calibrated_brier": cal_m["brier"],
            "accuracy": cal_m["accuracy"],
        })
    tbl = pd.DataFrame(table).sort_values("calibrated_log_loss").reset_index(drop=True)
    tbl.insert(0, "rank", np.arange(1, len(tbl) + 1))

    pd.set_option("display.width", 200)
    print("\n=== Σύγκριση, ίδιοι 76.583 γύροι ===\n")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # ---- paired tests of every Markov variant against plain OGD ----
    sig_rows = []
    base = per_round["OGD"]
    for label, v in per_round.items():
        if label == "OGD":
            continue
        d, n = paired_diff(v, base)
        r = moving_block_bootstrap_test(d)
        verdict = ("ισοπαλία" if not r["significant_95"]
                   else (f"{label} better" if r["mean_diff"] < 0 else "OGD better"))
        sig_rows.append({"series": label, "vs": "OGD", "n": n, "verdict": verdict, **r})
    sig = pd.DataFrame(sig_rows).sort_values("mean_diff")
    print("\n=== Έναντι του απλού OGD (αρνητικό = καλύτερο από OGD) ===\n")
    print(sig[["series", "n", "mean_diff", "ci_low", "ci_high", "p_value", "verdict"]]
          .to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    # ---- MCGD in the sense of Sun, Sun & Yin (2018): dependent sampling ----
    print("\n" + "=" * 70)
    print("MCGD κατά Sun, Sun & Yin (2018): το βήμα στον ΕΝΕΡΓΟ αριθμό δειγμάτων")
    print("=" * 70)
    b = int(round(T ** (1 / 3)))
    print(f"ορίζοντας εξάρτησης από το moving block bootstrap: b = T^(1/3) = {b}")
    print(f"θεωρητική διόγκωση βήματος sqrt(tau) = {np.sqrt(b):.2f}")
    print(f"διόγκωση από το χαλαρό φράγμα M: 13,82 / 3,76 = {13.82/3.76:.2f}")
    print(f"γινόμενο των δύο = {np.sqrt(b) * 13.82/3.76:.1f}   "
          f"(εμπειρικά βέλτιστο: {OGD_STEP_MULTIPLIER:.0f})\n")

    mc_rows = []
    for tau in (1, 5, 10, b, 100, 500):
        for mult in (1.0, 3.7):
            W = effective_sample_schedule(loss, awake, N, mult, tau)
            tr, va, _ = split_scores(W)
            mc_rows.append({"kernel": f"mcgd(tau={tau})", "alpha": mult,
                            "train": tr, "held_out": va})
            print(f"  τ={tau:<4} mult={mult:<4g} train {tr:.6f}  ουρά {va:.6f}")
    mc = pd.DataFrame(mc_rows)
    bmc = mc.loc[mc["train"].idxmin()]
    print(f"\n  καλύτερο στο train: {bmc['kernel']}, mult={bmc['alpha']:g}"
          f" -> ουρά {bmc['held_out']:.6f}")
    print(f"  ισχύον OGD (x{OGD_STEP_MULTIPLIER:.0f} σε παγκόσμιο t): ουρά "
          f"{sweep[sweep['kernel'] == 'identity']['held_out'].iloc[0]:.6f}")
    sweep = pd.concat([sweep, mc], ignore_index=True)

    sp = os.path.join(OUT_DIR, "markov_ogd.csv")
    sgp = os.path.join(OUT_DIR, "markov_ogd_significance.csv")
    pd.concat([sweep, tbl], ignore_index=True).to_csv(sp, index=False)
    sig.to_csv(sgp, index=False)
    print(f"\nSaved: {sp}\nSaved: {sgp}")
    plot_sweep(sweep, os.path.join(OUT_DIR, "markov_ogd.png"))


def plot_sweep(sweep, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    palette = {"uniform": COLORS["Hedge"], "pair": COLORS["OGD"],
               "phase": COLORS["FTRL"]}
    base = sweep[sweep["kernel"] == "identity"]["held_out"].iloc[0]
    ax.axhline(base, color="#898781", linestyle="--", linewidth=1.4,
               label="απλό OGD", zorder=1)
    for kind in ("uniform", "pair", "phase"):
        s = sweep[(sweep["kernel"] == kind) & (sweep["alpha"] > 0)]
        ax.plot(s["alpha"], s["held_out"], marker="o", markersize=5,
                color=palette[kind], linewidth=1.8, label=kind, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("ρυθμός μετάβασης α (λογαριθμικός άξονας)", color="#52514e")
    ax.set_ylabel("log-loss στην ουρά", color="#52514e")
    ax.set_title("Markov Chain OGD: τρεις πυρήνες μετάβασης",
                 color="#0b0b0b", fontsize=12, pad=12)
    _style_axes(ax)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color("#0b0b0b")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
