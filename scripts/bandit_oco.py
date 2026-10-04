"""
BANDIT ONLINE CONVEX OPTIMIZATION for the mixture weights.

Every OCO stage in this project so far is FULL INFORMATION: after each match the
learner is handed the whole per-expert loss vector loss[t, :] (every bookmaker's
own log-loss for that match) and steps along it. That is what
`run_ogd_sleeping` does, and it is only possible because we score all 24
bookmakers offline.

This file asks the harder question: how well can the same mixture be learned
when the ONLY thing observed each round is the scalar loss of the forecast the
learner itself published?

    f_t(w) = -log( <w, p_t(y_t)> )                          # one number, per round

That is the real, deployable loss of the mixture (the quantity every table in
`results/` reports), it is convex in w over the simplex, and it is the natural
feedback model for someone who records their own forecast's score but never
gets a per-bookmaker breakdown. It is strictly LESS information than OGD gets,
so the honest expectation is that bandit OCO loses; the question is by how
much, and that is what the comparison measures.

ALGORITHM -- Bandit Gradient Descent (Flaxman, Kalai & McMahan, 2005, "Online
convex optimization in the bandit setting: gradient descent without a
gradient"), i.e. OGD driven by a one-point gradient estimate built from a
random probe:

    y_t   in  Delta^xi          # shrunken simplex: y_i >= xi/n, room to probe
    u_t   ~  Unif(S^{d-1} cap {u : sum u_i = 0})     # probe in the TANGENT space
    w_t   =  y_t + delta u_t    # the point actually played
    g_t   =  (d / delta) * ( f_t(w_t) - b_t ) * u_t  # one-point estimate
    y_t+1 =  Pi_{Delta^xi}( y_t - eta_t g_t )

Three details that are specific to the simplex and easy to get wrong:

  * the simplex is (n-1)-dimensional, so the probe must live in the tangent
    space {sum u_i = 0} and the estimator's dimension factor is d = n-1, not n.
    Probing along a generic direction in R^n would leave the simplex outright;
  * delta = xi/n makes y_t +- delta u_t feasible for FREE: y_i >= xi/n and
    |u_i| <= 1, so every coordinate stays >= 0 and the sum stays exactly 1.
    One knob (xi) therefore controls both the shrink and the probe radius;
  * b_t (a running mean of the learner's OWN observed losses, so it uses no
    extra information) is pure variance reduction: E[c * u_t] = 0 for any c
    that does not depend on u_t, so subtracting it cannot bias the estimator.

The two-point variant (Agarwal, Dekel & Xiao, 2010) is also run: it probes
y_t +- delta u_t and uses (d / 2delta) * (f(+) - f(-)) * u_t. Still scalar
function-value feedback only -- no per-expert losses ever -- but two queries of
the same round's loss instead of one, which cuts the variance enormously. It is
a genuinely stronger assumption and is labelled as such everywhere.

SLEEPING EXPERTS: identical reduction to `sleeping_experts.py` -- the probe, the
estimator, the projection and the update all live on the awake subset A(t)
alone, so n = |A(t)| and d = n-1 vary round to round, and asleep coordinates
stay frozen. A round with n = 1 has d = 0, i.e. no direction to probe in: the
(forced) weight 1.0 is played and nothing is learned.

WHAT IS COMPARED, and the one accounting subtlety. Two numbers come out of a
bandit run and they answer different questions:

  * `played`  -- the loss of the point actually published, y_t + delta u_t. This
    is what the bandit learner truly pays, exploration included, and it is the
    quantity its regret bound is about;
  * `center`  -- the loss of y_t, the un-probed state. This is the forecast one
    would deploy, and it is the only one comparable with OGD on the full metric
    battery (Brier/RPS/accuracy, and calibration). Evaluating it is an
    ANALYST's measurement: the algorithm never observes it and never uses it.

Both are reported. The headline comparison against OGD uses `center`, with
`played` alongside so the exploration cost is visible rather than hidden.

TWO CONTROLS, because "BGD vs OGD" otherwise conflates two separate changes:
  1. `full_exact` -- full-information OGD on the exact gradient of the SAME
     non-linear loss the bandit sees, -p/<w,p>. Isolates "bandit vs full
     information" from "non-linear loss vs the project's linearisation".
  2. `full_linear` -- the project's shipped linearisation, asserted to
     reproduce `run_ogd_sleeping` EXACTLY (see CLAUDE.md: any variant of these
     algorithms must carry that check).

SELECTION: xi and the step multiplier are chosen on the chronological train
prefix only, in two stages (step profile at a middle xi, then xi x a local step
window) -- the same protocol and the same two-stage reasoning as every other
hyperparameter here. `_extend_if_endpoint` enforces the project's hard-learned
rule that a winner sitting on a grid endpoint is evidence the grid is too
narrow, not evidence of an optimum. The algorithm is RANDOMISED, so the
selected configuration is re-run across SEEDS and the spread is reported; a
single seed's number would be noise.

Produces:
  - results/bandit_oco.csv
  - results/bandit_oco_significance.csv
  - results/bandit_oco.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from sleeping_experts import (
    OUT_DIR, EPS, M as LOSS_BOUND, COLORS,
    load_full_universe, per_round_expert_loss, project_to_simplex,
    run_ogd_sleeping, evaluate, OGD_STEP_MULTIPLIER,
)
from calibration_analysis import _style_axes
from final_ranking import calibrate_full_history
from significance_test import moving_block_bootstrap_test, paired_diff

TRAIN_FRACTION = 0.75

# The bandit estimator's magnitude is ~(d/delta) ~ 5e3 times a full-information
# gradient's, so the useful step multiplier lives many decades below OGD's 100.
# The grid spans 9 decades for that reason; _extend_if_endpoint widens it
# further if the winner still lands on an edge.
ETA_GRID = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
XI_GRID = (0.005, 0.02, 0.05, 0.1, 0.25, 0.5)
XI_PROFILE = 0.05                 # stage-A xi, an interior point of XI_GRID
ESTIMATORS = ("two_point", "one_point")
BATCH_GRID = (1, 20, 100, 500)
BATCH_ETA_GRID = (1e-6, 1e-4, 1e-2, 1.0, 100.0)  # wide: batching rescales the step
SEEDS = tuple(range(10))
SWEEP_SEED = 0
MAX_EXTENSIONS = 4


# --------------------------------------------------------------- geometry --

def project_to_shrunk_simplex(v, xi):
    """Euclidean projection onto {w : w_i >= xi/n, sum w = 1}.

    That set is exactly xi/n * 1 + (1 - xi) * Delta, so the projection is the
    plain simplex projection of the rescaled point, mapped back -- the
    (1-xi)^2 factor in the objective does not move the argmin.

    xi = 0 returns `project_to_simplex` UNCHANGED (not merely equal to it),
    which is what lets the full-information modes reproduce
    `run_ogd_sleeping` bit for bit rather than approximately.
    """
    if xi <= 0.0:
        return project_to_simplex(v)
    floor = xi / len(v)
    return floor + (1.0 - xi) * project_to_simplex((v - floor) / (1.0 - xi))


def tangent_probe(g):
    """Unit vector in the simplex's tangent space {sum u_i = 0}, from a raw
    Gaussian draw. Returns None when no direction exists (n = 1, so d = 0)."""
    u = g - g.mean()
    nrm = np.linalg.norm(u)
    if nrm < 1e-12:
        return None
    return u / nrm


# -------------------------------------------------------------- algorithm --

def run_bandit_ogd(q, awake, N, mode, xi=0.0, eta_mult=OGD_STEP_MULTIPLIER,
                   seed=0, baseline=True, loss=None, batch=1):
    """OGD on the mixture weights under one of four information models.

    q      : (T, N) the probability each expert put on the REALISED outcome,
             i.e. P[t, :, y[t]]. The bandit modes touch it only through the
             scalar queries `-log(<w, q[t, awake]>)` -- never per coordinate,
             which is what makes the information model bandit rather than full.
    mode   : "full_linear" (the project's shipped OGD -- asserted identical),
             "full_exact"  (exact gradient of the non-linear loss),
             "one_point"   (Flaxman/Kalai/McMahan 2005),
             "two_point"   (Agarwal/Dekel/Xiao 2010).
    xi     : shrink of the simplex; also fixes the probe radius delta = xi/n.
             Ignored (forced to 0) by the full-information modes.
    loss   : (T, N) per-expert log-loss, required by "full_linear" only.

    Returns (W_center, played, n_queries):
      W_center (T, N) the un-probed state y_t, renormalised over A(t) -- the
        deployable forecast, written only on awake coordinates (so read it with
        `sleeping_experts.last_awake_weight`, never as W[-1]);
      played   (T,)   per-round loss of the point actually published, NaN on a
        round where nobody was awake;
      n_queries       how many scalar loss evaluations the algorithm consumed.
    """
    if mode not in ("full_linear", "full_exact", "one_point", "two_point"):
        raise ValueError(mode)
    bandit = mode in ("one_point", "two_point")
    if mode == "full_linear" and loss is None:
        raise ValueError("full_linear needs the per-expert loss matrix")
    xi_eff = xi if bandit else 0.0

    D = np.sqrt(2.0)
    T = q.shape[0]
    w = np.full(N, 1.0 / N)           # persistent state, frozen while asleep
    W = np.zeros((T, N))
    played = np.full(T, np.nan)
    n_queries = 0
    b, n_b = 0.0, 0                   # running-mean baseline over observed losses

    rng = np.random.default_rng(seed)
    # one allocation instead of T small ones; column j of row t is i.i.d. N(0,1)
    G = rng.standard_normal((T, N)) if bandit else None

    # MINI-BATCHING (bandit modes only; batch=1 is the ungrouped original).
    # Motivation, measured rather than assumed: the one-point estimate's mean
    # cosine with the true gradient is +0.002, so the signal is 0.2% of its
    # magnitude. Averaging over T rounds cuts noise by sqrt(T) ~ 277, leaving an
    # effective ratio near 0.55 -- still below one, which is why every positive
    # step hurts and the train selector drives eta to zero. Accumulating the
    # estimate over `batch` awake rounds before stepping raises that ratio by
    # sqrt(batch). It also keeps the iterate in the SIMPLEX INTERIOR: a single
    # raw one-point step has norm ~(d/delta)|f-b| ~ 2000, which the projection
    # clips asymmetrically, and that clipping is itself a bias.
    # Coordinates asleep for the whole batch accumulate exactly zero, so the
    # Sleeping Experts freezing property is preserved.
    g_acc = np.zeros(N)
    n_acc = 0

    for t in range(T):
        a = awake[t]
        if not a.any():
            continue                  # nobody quoting: nothing to play, nothing to learn
        n = int(a.sum())
        v = project_to_shrunk_simplex(w[a], xi_eff)
        W[t, a] = v
        w[a] = v
        qa = q[t, a]
        eta = eta_mult * D / (LOSS_BOUND * np.sqrt(t + 1))

        if mode == "full_linear":
            played[t] = -np.log(max(float(v @ qa), EPS))
            g = loss[t, a]
        elif mode == "full_exact":
            mix = max(float(v @ qa), EPS)
            played[t] = -np.log(mix)
            g = -qa / mix
        else:
            d = n - 1
            u = tangent_probe(G[t, :n]) if d > 0 else None
            if u is None:
                # forced weight, no direction to probe in: play and learn nothing
                played[t] = -np.log(max(float(v @ qa), EPS))
                n_queries += 1
                continue
            delta = xi / n
            f_plus = -np.log(max(float((v + delta * u) @ qa), EPS))
            n_queries += 1
            if mode == "one_point":
                played[t] = f_plus
                g = (d / delta) * (f_plus - (b if baseline else 0.0)) * u
                n_b += 1
                b += (f_plus - b) / n_b
            else:
                f_minus = -np.log(max(float((v - delta * u) @ qa), EPS))
                n_queries += 1
                played[t] = 0.5 * (f_plus + f_minus)     # pays for both probes
                g = (d / (2.0 * delta)) * (f_plus - f_minus) * u

        if not bandit or batch <= 1:
            w[a] = w[a] - eta * g
        else:
            g_acc[a] += g
            n_acc += 1
            if n_acc >= batch:
                w -= eta * (g_acc / n_acc)
                g_acc[:] = 0.0
                n_acc = 0
    return W, played, n_queries


# --------------------------------------------------------------- sanity -----

def check_estimator_unbiased(q, awake, rng, n_draws=200000):
    """The two-point estimate must average to the TANGENTIAL part of the true
    gradient: for u uniform on the unit sphere of a d-dimensional subspace,
    E[u u^T] = P / d, so E[d <grad f, u> u] = P grad f.

    This is the bandit analogue of the assertions the other variant files carry
    -- `full_linear` pins the update RULE to the shipped algorithm, and this
    pins the gradient ESTIMATOR to the gradient it is meant to estimate.
    Vectorised over draws, so it can afford enough of them for the Monte-Carlo
    error to sit well below the tolerance. Returns (relative error, cosine)."""
    n_aw = awake.sum(axis=1)
    target = min(10, int(n_aw.max()))          # the real panel reaches 18; a
    assert target >= 3, "need >=3 awake experts for a meaningful probe"
    wide = np.flatnonzero(n_aw >= target)      # truncated panel may not
    t = int(wide[len(wide) // 3])
    a = awake[t]
    n = int(a.sum())
    d = n - 1
    qa = q[t, a]
    v = project_to_shrunk_simplex(np.full(n, 1.0 / n), 0.1)
    delta = 1e-4

    grad = -qa / float(v @ qa)
    tangential = grad - grad.mean()                    # projection onto {sum u = 0}

    U = rng.standard_normal((n_draws, n))
    U -= U.mean(axis=1, keepdims=True)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    f_p = -np.log(np.clip((v + delta * U) @ qa, EPS, None))
    f_m = -np.log(np.clip((v - delta * U) @ qa, EPS, None))
    est = ((((d / (2.0 * delta)) * (f_p - f_m))[:, None]) * U).mean(axis=0)

    rel = float(np.linalg.norm(est - tangential) / np.linalg.norm(tangential))
    cos = float(est @ tangential / (np.linalg.norm(est) * np.linalg.norm(tangential)))
    return rel, cos


def estimator_alignment(q, awake, mode, xi, rng, n_rounds=4000, baseline_value=1.0):
    """Mean cosine between a SINGLE-draw estimate at the operating point and the
    true tangential gradient, over a sample of rounds.

    Unbiasedness (above) says the estimates average to the gradient. It says
    nothing about how much of any one of them is signal, and that is what
    decides whether 76,583 rounds are enough. This measures it directly:

      two-point:  est = d <grad, u> u, always on the correct side of the
                  gradient, so the cosine is positive and of order 1/sqrt(d);
      one-point:  est = (d/delta)(f - b) u, dominated by a term pointing in a
                  direction that has nothing to do with the gradient, so the
                  cosine sits near zero and the algorithm survives only by
                  averaging over the whole history.

    Returns (mean cosine, std of cosine)."""
    n_aw = awake.sum(axis=1)
    cand = np.flatnonzero(n_aw >= 3)
    idx = cand[np.linspace(0, len(cand) - 1, min(n_rounds, len(cand))).astype(int)]
    cos = []
    for t in idx:
        a = awake[t]
        n = int(a.sum())
        d = n - 1
        u = tangent_probe(rng.standard_normal(n))
        if u is None:
            continue
        qa = q[t, a]
        v = project_to_shrunk_simplex(np.full(n, 1.0 / n), xi)
        mix = float(v @ qa)
        grad = -qa / mix
        tang = grad - grad.mean()
        if np.linalg.norm(tang) < 1e-12:
            continue
        delta = xi / n
        f_p = -np.log(max(float((v + delta * u) @ qa), EPS))
        if mode == "two_point":
            f_m = -np.log(max(float((v - delta * u) @ qa), EPS))
            est = (d / (2.0 * delta)) * (f_p - f_m) * u
        else:
            # the running baseline converges to the mean observed loss, which on
            # this data is ~1.0; using that constant keeps the diagnostic
            # independent of where in the history the sampled round sits
            est = (d / delta) * (f_p - baseline_value) * u
        nrm = np.linalg.norm(est)
        if nrm < 1e-300:
            continue
        cos.append(float(est @ tang / (nrm * np.linalg.norm(tang))))
    cos = np.array(cos)
    return float(cos.mean()), float(cos.std(ddof=1))


# ----------------------------------------------------------------- sweep ----

def _extend_if_endpoint(values, evaluate_one, scores, max_ext=MAX_EXTENSIONS,
                        bounds=(0.0, np.inf)):
    """Keep widening a geometric grid while its best point sits on an edge.

    The project's most expensive methodological lesson (see
    learning_rate_study.py and CLAUDE.md): a sweep whose answer lands on a grid
    boundary has not found an optimum, it has found the boundary. Hedge and
    FTRL were both reported "already optimal" at what was simply the smallest
    value anyone had tried.

    `values` and `scores` are extended in place; `evaluate_one(x)` returns the
    train score for a new candidate. `bounds` stops the walk where the next
    candidate would be meaningless (xi must stay strictly inside (0, 1)).
    Returns True if the best point is STILL on an edge when it gives up, which
    the caller should say out loud.
    """
    for _ in range(max_ext):
        i = int(np.argmin(scores))
        if 0 < i < len(values) - 1:
            return False
        step = values[1] / values[0] if i == 0 else values[-1] / values[-2]
        x = values[0] / step if i == 0 else values[-1] * step
        if not (bounds[0] < x < bounds[1]):
            print(f"      ! ακμή του grid στο {values[i]:g}, αλλά το {x:g} "
                  f"είναι εκτός ορίων {bounds} -- σταματώ")
            return True
        print(f"      ! ακμή του grid: επεκτείνω σε {x:g}")
        s = evaluate_one(x)
        if i == 0:
            values.insert(0, x)
            scores.insert(0, s)
        else:
            values.append(x)
            scores.append(s)
    return True


def sweep_estimator(q, awake, N, cut, mode, rows):
    """Two-stage train-only selection of (xi, eta_mult) for one estimator.

    Two stages rather than one joint grid for the reason the project already
    established for the step x calibration-rate pair: the two knobs turned out
    not to interact, and a joint grid large enough to see that costs far more
    runs than it earns. `seen` caches (xi, eta) so stage B never re-runs a
    configuration stage A already measured -- keyed per estimator, since `rows`
    is shared across them."""
    seen = {}

    def score(xi, eta, seed=SWEEP_SEED, tag="sweep"):
        key = (round(xi, 12), round(eta, 12))
        if key in seen:
            return seen[key]
        W, played, _ = run_bandit_ogd(q, awake, N, mode, xi=xi, eta_mult=eta, seed=seed)
        centre = center_loss(W, q)
        tr, ho = np.nanmean(centre[:cut]), np.nanmean(centre[cut:])
        rows.append({"stage": tag, "estimator": mode, "xi": xi, "eta_mult": eta,
                     "seed": seed, "train": tr, "held_out": ho,
                     "played_train": np.nanmean(played[:cut]),
                     "played_held_out": np.nanmean(played[cut:])})
        print(f"    xi={xi:<5g} eta={eta:<8g} train {tr:.6f}  ουρά {ho:.6f}"
              f"  (played ουρά {np.nanmean(played[cut:]):.6f})")
        seen[key] = tr
        return tr

    print(f"\n  -- στάδιο A: προφίλ βήματος στο xi={XI_PROFILE} --")
    etas = list(ETA_GRID)
    scores = [score(XI_PROFILE, e, tag="A") for e in etas]
    if _extend_if_endpoint(etas, lambda x: score(XI_PROFILE, x, tag="A"), scores):
        print("      ! ΠΡΟΣΟΧΗ: το βήμα παραμένει στην ακμή μετά τις επεκτάσεις")
    eta_1 = etas[int(np.argmin(scores))]
    print(f"    -> βήμα σταδίου A: {eta_1:g}")

    print(f"\n  -- στάδιο B: σάρωση xi στο eta={eta_1:g} --")
    xis = list(XI_GRID)
    xs = [score(x, eta_1, tag="B") for x in xis]
    # xi is a shrink AND the probe radius, so it must stay strictly inside (0,1):
    # at xi -> 1 the feasible set collapses to the uniform point.
    if _extend_if_endpoint(xis, lambda x: score(x, eta_1, tag="B"), xs,
                           bounds=(1e-4, 0.9)):
        print("      ! ΠΡΟΣΟΧΗ: το xi παραμένει στην ακμή μετά τις επεκτάσεις")
    xi_star = xis[int(np.argmin(xs))]
    print(f"    -> xi σταδίου B: {xi_star:g}")

    print(f"\n  -- στάδιο C: επανεκλέπτυνση βήματος στο xi={xi_star:g} --")
    fine = sorted(eta_1 * r for r in (1 / 9, 1 / 3, 1.0, 3.0, 9.0))
    fs = [score(xi_star, e, tag="C") for e in fine]
    if _extend_if_endpoint(fine, lambda x: score(xi_star, x, tag="C"), fs):
        print("      ! ΠΡΟΣΟΧΗ: το βήμα παραμένει στην ακμή μετά τις επεκτάσεις")
    eta_star = fine[int(np.argmin(fs))]
    print(f"    -> επιλογή στο train: xi={xi_star:g}, eta={eta_star:g} "
          f"(train {min(fs):.6f})")
    return xi_star, eta_star


def center_loss(W, q):
    """Per-round log-loss of the un-probed state, NaN where no expert was awake."""
    mix = (W * q).sum(axis=1)
    out = np.full(len(mix), np.nan)
    live = mix > 0.0
    out[live] = -np.log(np.clip(mix[live], EPS, 1.0))
    return out


# ------------------------------------------------------------------ main ----

def main():
    panel, P, awake, y, dates, seasons = load_full_universe()
    T, N, _ = P.shape
    loss = per_round_expert_loss(P, y)
    q = P[np.arange(T), :, y]                  # (T, N) prob on the realised outcome
    cut = int(round(T * TRAIN_FRACTION))
    n_awake = awake.sum(axis=1)
    print(f"{T} γύροι, {N} experts, awake/γύρο: min {n_awake.min()}, "
          f"μέσος {n_awake.mean():.2f}, max {n_awake.max()}")
    print(f"train ως τον γύρο {cut}, ουρά {T - cut}")
    print(f"γύροι με d=0 (ένας μόνο awake, τίποτα να εξερευνηθεί): "
          f"{int((n_awake == 1).sum())}\n")

    # ---- sanity 1: the shipped algorithm must come out of this code path ----
    W_lin, played_lin, nq_lin = run_bandit_ogd(q, awake, N, "full_linear", loss=loss)
    assert np.allclose(W_lin, run_ogd_sleeping(loss, awake, N)), \
        "full_linear must reproduce run_ogd_sleeping exactly"
    assert nq_lin == 0, "a full-information mode must consume no bandit queries"
    print("sanity ok: mode='full_linear' == run_ogd_sleeping, 0 queries")

    # ---- sanity 2: the gradient ESTIMATOR must estimate the gradient --------
    rel, cos = check_estimator_unbiased(q, awake, np.random.default_rng(7))
    print(f"sanity ok: two-point estimator vs true tangential gradient -- "
          f"σχετικό σφάλμα {rel:.4f}, cos {cos:.6f}")
    assert rel < 0.05 and cos > 0.999, "two-point estimator is not tracking the gradient"

    # ---- the full-information control gets its step tuned the same way ------
    # Otherwise "bandit vs full information" would compare a tuned algorithm
    # against an untuned one. `full_linear` is already tuned (that IS
    # OGD_STEP_MULTIPLIER); `full_exact` steps along a different gradient, so
    # it needs its own selection on the same train prefix.
    print("\n-- βήμα για το full_exact control, επιλογή στο train --")
    rows = []
    ex_etas, ex_scores = list(ETA_GRID), []

    def score_exact(eta):
        W, _, _ = run_bandit_ogd(q, awake, N, "full_exact", eta_mult=eta)
        c = center_loss(W, q)
        tr, ho = np.nanmean(c[:cut]), np.nanmean(c[cut:])
        rows.append({"stage": "A_exact", "estimator": "full_exact", "xi": 0.0,
                     "eta_mult": eta, "seed": -1, "train": tr, "held_out": ho})
        print(f"    eta={eta:<8g} train {tr:.6f}  ουρά {ho:.6f}")
        return tr

    ex_scores = [score_exact(e) for e in ex_etas]
    if _extend_if_endpoint(ex_etas, score_exact, ex_scores):
        print("      ! ΠΡΟΣΟΧΗ: παραμένει στην ακμή μετά τις επεκτάσεις")
    eta_exact = ex_etas[int(np.argmin(ex_scores))]
    print(f"    -> full_exact eta={eta_exact:g}")

    # ---- references ---------------------------------------------------------
    W_exact, played_exact, _ = run_bandit_ogd(q, awake, N, "full_exact",
                                              eta_mult=eta_exact)
    denom = n_awake[:, None]
    W_unif = np.divide(awake, denom, out=np.zeros_like(awake, dtype=float),
                       where=denom > 0)
    ref_ho = {"OGD (full info, linearised)": np.nanmean(center_loss(W_lin, q)[cut:]),
              "OGD (full info, exact grad)": np.nanmean(center_loss(W_exact, q)[cut:]),
              "Uniform average": np.nanmean(center_loss(W_unif, q)[cut:])}
    print("\nΑναφορές, log-loss στην ουρά:")
    for k, v in ref_ho.items():
        print(f"  {k:32s} {v:.6f}")

    # ---- selection, per estimator ------------------------------------------
    chosen = {}
    for mode in ESTIMATORS:
        print(f"\n{'=' * 72}\n{mode}\n{'=' * 72}")
        chosen[mode] = sweep_estimator(q, awake, N, cut, mode, rows)

    # ---- does mini-batching rescue the one-point estimator? ----------------
    # Asked because the three-stage selection above drove one-point's step to
    # ~1e-14: the whole region 1e-14..9e-11 ties at train 0.998375, i.e. it is
    # the no-learning plateau and the selector is simply taking its boundary.
    # Batching is the one change the variance argument actually licenses, so it
    # is tried explicitly and reported whatever it gives.
    print(f"\n{'=' * 72}\nmini-batching για το one_point (ξ={chosen['one_point'][0]:g})"
          f"\n{'=' * 72}")
    xi1 = chosen["one_point"][0]
    best_batch = (np.inf, 1, 0.0)
    for bsz in BATCH_GRID:
        for eta in BATCH_ETA_GRID:
            W, played, _ = run_bandit_ogd(q, awake, N, "one_point", xi=xi1,
                                          eta_mult=eta, batch=bsz, seed=SWEEP_SEED)
            c = center_loss(W, q)
            tr, ho = float(np.nanmean(c[:cut])), float(np.nanmean(c[cut:]))
            rows.append({"stage": "batch", "estimator": "one_point", "xi": xi1,
                         "eta_mult": eta, "batch": bsz, "seed": SWEEP_SEED,
                         "train": tr, "held_out": ho})
            print(f"    batch={bsz:<5d} eta={eta:<8g} train {tr:.6f}  ουρά {ho:.6f}")
            if tr < best_batch[0]:
                best_batch = (tr, bsz, eta)
    _, batch_star, eta_batch = best_batch
    print(f"  -> επιλογή στο train: batch={batch_star}, eta={eta_batch:g} "
          f"(train {best_batch[0]:.6f})")

    # ---- the selected configurations across seeds ---------------------------
    print(f"\n{'=' * 72}\nΕπιλεγμένες ρυθμίσεις σε {len(SEEDS)} seeds\n{'=' * 72}")
    seed_rows, best_seed = [], {}
    for mode in ESTIMATORS:
        xi, eta = chosen[mode]
        hos, runs = [], {}
        for s in SEEDS:
            W, played, nq = run_bandit_ogd(q, awake, N, mode, xi=xi, eta_mult=eta, seed=s)
            centre = center_loss(W, q)
            ho = np.nanmean(centre[cut:])
            hos.append(ho)
            runs[s] = (W, played, centre)
            seed_rows.append({"stage": "seeds", "estimator": mode, "xi": xi,
                              "eta_mult": eta, "seed": s,
                              "train": np.nanmean(centre[:cut]), "held_out": ho,
                              "played_train": np.nanmean(played[:cut]),
                              "played_held_out": np.nanmean(played[cut:]),
                              "queries_per_round": nq / T})
        hos = np.array(hos)
        med = SEEDS[int(np.argsort(hos)[len(hos) // 2])]
        best_seed[mode] = (med, runs[med])
        print(f"  {mode:10s} xi={xi:g} eta={eta:g}: ουρά μέσος {hos.mean():.6f} "
              f"± {hos.std(ddof=1):.6f}  [{hos.min():.6f}, {hos.max():.6f}]"
              f"  -> διάμεσο seed {med}")

    # ---- does the variance-reduction baseline earn its place? --------------
    print("\nΈλεγχος baseline (one_point, ίδιο seed):")
    xi, eta = chosen["one_point"]
    for use_b in (True, False):
        W, _, _ = run_bandit_ogd(q, awake, N, "one_point", xi=xi, eta_mult=eta,
                                 seed=SWEEP_SEED, baseline=use_b)
        c = center_loss(W, q)
        print(f"  baseline={str(use_b):5s} train {np.nanmean(c[:cut]):.6f}  "
              f"ουρά {np.nanmean(c[cut:]):.6f}")
        seed_rows.append({"stage": "baseline_check", "estimator": "one_point",
                          "xi": xi, "eta_mult": eta, "seed": SWEEP_SEED,
                          "train": np.nanmean(c[:cut]), "held_out": np.nanmean(c[cut:]),
                          "baseline": use_b})

    # ---- how much of one round's estimate is actually signal? ---------------
    print("\nΕυθυγράμμιση μιας εκτίμησης με την αληθινή κλίση (cosine ανά γύρο,"
          " στο επιλεγμένο ξ):")
    for mode in ESTIMATORS:
        xi, eta = chosen[mode]
        m, s = estimator_alignment(q, awake, mode, xi, np.random.default_rng(11))
        print(f"  {mode:10s} ξ={xi:<9g} μέσο cos {m:+.4f}  (sd {s:.4f})")
        seed_rows.append({"stage": "alignment", "estimator": mode, "xi": xi,
                          "eta_mult": eta, "seed": 11,
                          "mean_cosine": m, "sd_cosine": s})

    # ---- full-history comparison, IDENTICAL round set for every series -----
    entries = {"OGD (full info, linearised)": W_lin,
               "OGD (full info, exact grad)": W_exact}
    for mode in ESTIMATORS:
        xi, eta = chosen[mode]
        label = f"BGD {mode.replace('_', '-')} (xi={xi:g})"
        entries[label] = best_seed[mode][1][0]

    # ---- THE CONTROL THAT MAKES THE BGD NUMBERS INTERPRETABLE --------------
    # Under Sleeping Experts the weight vector moves even with NO learning at
    # all: every time the awake set changes, the persistent state is re-projected
    # onto a simplex of a different dimension, and that churn alone produces a
    # non-uniform, better-than-uniform mixture. Measured directly: at eta = 0 the
    # weights still end up 0.69 away from uniform in max norm.
    #
    # So "BGD beats the uniform average" is NOT evidence that bandit feedback
    # taught it anything -- most or all of that gap can be projection churn. The
    # honest baseline is the same algorithm with the step set to zero: identical
    # shrink, identical probing, identical projection dynamics, zero learning.
    # Everything between the control and BGD is what the bandit signal bought.
    for mode in ESTIMATORS:
        xi, _ = chosen[mode]
        W0, _, _ = run_bandit_ogd(q, awake, N, mode, xi=xi, eta_mult=0.0, seed=0)
        entries[f"Χωρίς μάθηση (η=0, ξ={xi:g})"] = W0
    if batch_star > 1:
        Wb, _, _ = run_bandit_ogd(q, awake, N, "one_point", xi=xi1,
                                  eta_mult=eta_batch, batch=batch_star,
                                  seed=SWEEP_SEED)
        entries[f"BGD one-point, batch={batch_star}"] = Wb
    entries["Uniform average"] = W_unif

    alive = np.ones(T, dtype=bool)
    table, per_round_cal, per_round_raw = [], {}, {}
    for label, W in entries.items():
        phat = np.einsum("tn,tnk->tk", W, P)
        p, cal, yy, _ = calibrate_full_history(phat, y, alive)
        raw_m, cal_m = evaluate(p, yy), evaluate(cal, yy)
        per_round_cal[label] = -np.log(np.clip(cal[np.arange(T), yy], EPS, 1.0))
        per_round_raw[label] = -np.log(np.clip(p[np.arange(T), yy], EPS, 1.0))
        row = {"name": label,
               "raw_log_loss": raw_m["log_loss"],
               "calibrated_log_loss": cal_m["log_loss"],
               "raw_brier": raw_m["brier"], "accuracy": cal_m["accuracy"],
               "held_out_raw_log_loss": np.nanmean(center_loss(W, q)[cut:])}
        if label.startswith("BGD"):
            mode = "two_point" if "two-point" in label else "one_point"
            row["played_log_loss"] = np.nanmean(best_seed[mode][1][1])
        table.append(row)
    tbl = pd.DataFrame(table).sort_values("calibrated_log_loss").reset_index(drop=True)
    tbl.insert(0, "rank", np.arange(1, len(tbl) + 1))

    # every series is defined on all T rounds here -- check it rather than assume
    for label, v in per_round_cal.items():
        assert np.isfinite(v).all(), f"{label} is not defined on every round"

    pd.set_option("display.width", 220)
    print(f"\n=== Σύγκριση, ίδιοι {T} γύροι ===\n")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    # ---- decomposition: churn vs learning vs the information gap ------------
    get = lambda nm: float(tbl.loc[tbl["name"] == nm, "calibrated_log_loss"].iloc[0])
    unif, ogd = get("Uniform average"), get("OGD (full info, linearised)")
    print("\n=== Τι αγοράζει τι (βαθμονομημένο log-loss) ===")
    print(f"  ομοιόμορφος μέσος {unif:.6f}   ->   OGD πλήρους πληροφόρησης "
          f"{ogd:.6f}   (συνολικό περιθώριο {unif - ogd:.6f})")
    for mode in ESTIMATORS:
        xi, eta = chosen[mode]
        ctrl = get(f"Χωρίς μάθηση (η=0, ξ={xi:g})")
        bgd = get(f"BGD {mode.replace('_', '-')} (xi={xi:g})")
        churn, learned = unif - ctrl, ctrl - bgd
        print(f"\n  {mode}  (επιλεγμένο η={eta:g})")
        print(f"    προβολή χωρίς μάθηση : {unif:.6f} -> {ctrl:.6f}  "
              f"({churn:+.6f}, {100 * churn / (unif - ogd):5.1f}% του περιθωρίου)")
        print(f"    + bandit μάθηση      : {ctrl:.6f} -> {bgd:.6f}  "
              f"({learned:+.6f}, {100 * learned / (unif - ogd):5.1f}%)")
        if abs(learned) < 1e-6:
            print("    ! η μάθηση δεν προσθέτει τίποτα μετρήσιμο: ό,τι φαίνεται ως "
                  "κέρδος έναντι του ομοιόμορφου είναι τεχνούργημα της προβολής")

    # ---- paired tests against the shipped OGD ------------------------------
    base_label = "OGD (full info, linearised)"
    sig_rows = []
    for stage, series in (("calibrated", per_round_cal), ("raw", per_round_raw)):
        base = series[base_label]
        for label, v in series.items():
            if label == base_label:
                continue
            d, n = paired_diff(v, base)
            r = moving_block_bootstrap_test(d)
            verdict = ("ισοπαλία" if not r["significant_95"]
                       else (f"{label} better" if r["mean_diff"] < 0 else "OGD better"))
            sig_rows.append({"stage": stage, "series": label, "vs": base_label,
                             "n": n, "verdict": verdict, **r})
    # and the honest bandit accounting: played loss vs OGD's own loss
    for mode in ESTIMATORS:
        played = best_seed[mode][1][1]
        d, n = paired_diff(played, played_lin)
        r = moving_block_bootstrap_test(d)
        sig_rows.append({"stage": "played_raw", "series": f"BGD {mode} (played)",
                         "vs": "OGD (played)", "n": n,
                         "verdict": ("ισοπαλία" if not r["significant_95"]
                                     else ("BGD better" if r["mean_diff"] < 0
                                           else "OGD better")), **r})
    sig = pd.DataFrame(sig_rows)
    print("\n=== Έναντι του ισχύοντος OGD (αρνητικό mean_diff = καλύτερο από OGD) ===\n")
    print(sig[["stage", "series", "n", "mean_diff", "ci_low", "ci_high",
               "p_value", "verdict"]].to_string(index=False,
                                                float_format=lambda v: f"{v:.6f}"))

    sweep = pd.DataFrame(rows + seed_rows)
    sp = os.path.join(OUT_DIR, "bandit_oco.csv")
    sgp = os.path.join(OUT_DIR, "bandit_oco_significance.csv")
    pd.concat([sweep, tbl], ignore_index=True).to_csv(sp, index=False)
    sig.to_csv(sgp, index=False)
    print(f"\nSaved: {sp}\nSaved: {sgp}")
    plot_bandit(sweep, tbl, ref_ho, os.path.join(OUT_DIR, "bandit_oco.png"))


# ------------------------------------------------------------------ plot ----

BANDIT_COLORS = {                      # project palette only -- see CLAUDE.md
    "two_point": COLORS["Hedge"],
    "one_point": COLORS["FTRL"],
    "OGD (full info, linearised)": COLORS["OGD"],
    "OGD (full info, exact grad)": COLORS["benchmark"],
    "Uniform average": COLORS["Uniform average"],
}
# Checked with the dataviz skill's validator against this surface, so it does
# not need re-deriving: the four categorical hues (#2a78d6, #1baf7a, #eb6834,
# #4a3aa7) pass every check -- worst adjacent CVD separation dE 9.2 (deutan),
# normal-vision floor 24.0. COLORS["benchmark"] (#52514e) fails the chroma
# floor BY DESIGN: it is the project's deliberate neutral for a reference
# series, not a fifth categorical hue. The one WARN (#1baf7a contrast 2.74:1)
# is relieved as the skill requires -- every dot carries its value as a direct
# label, the line panel carries a legend plus markers/linestyle, and
# bandit_oco.csv is the table view.
INK, INK_SOFT, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def plot_bandit(sweep, tbl, ref_ho, out_path):
    """Left: the step profile that selection is based on. Right: a dot plot of
    calibrated log-loss (dots, not bars -- these cluster near 1.0 and have no
    meaningful zero, so a truncated baseline would exaggerate the gaps)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)

    prof = sweep[sweep["stage"] == "A"]
    for mode in ESTIMATORS:
        s = prof[prof["estimator"] == mode].sort_values("eta_mult")
        ax1.plot(s["eta_mult"], s["held_out"], marker="o", markersize=6,
                 linewidth=2.0, color=BANDIT_COLORS[mode],
                 label=f"BGD {mode.replace('_', '-')}", zorder=3)
    for label, style in (("OGD (full info, linearised)", "-"),
                         ("Uniform average", "--")):
        ax1.axhline(ref_ho[label], color=BANDIT_COLORS[label], linestyle=style,
                    linewidth=1.6, zorder=2, label=label)
    ax1.set_xscale("log")
    ax1.set_xlabel(f"πολλαπλασιαστής βήματος (ξ={XI_PROFILE}, λογαριθμικός άξονας)",
                   color=INK_SOFT)
    ax1.set_ylabel("log-loss στην ουρά", color=INK_SOFT)
    ax1.set_title("Προφίλ βήματος: bandit vs πλήρης πληροφόρηση",
                  color=INK, fontsize=12, pad=12)
    ax1.set_ymargin(0.10)
    _style_axes(ax1)
    # the curves occupy the top-left and the reference lines span the full
    # width, so the mid-left band is the only reliably empty region
    leg = ax1.legend(frameon=False, fontsize=8.5, loc="center left")
    for t in leg.get_texts():
        t.set_color(INK)

    d = tbl.sort_values("calibrated_log_loss", ascending=False)
    ypos = np.arange(len(d))
    lo, hi = d["calibrated_log_loss"].min(), d["calibrated_log_loss"].max()
    span = max(hi - lo, 1e-9)
    for yv, (_, r) in zip(ypos, d.iterrows()):
        key = ("two_point" if "two-point" in r["name"]
               else "one_point" if "one-point" in r["name"] else r["name"])
        col = BANDIT_COLORS.get(key, COLORS["muted"])
        ax2.hlines(yv, lo, r["calibrated_log_loss"],
                   color=col, alpha=0.28, linewidth=2.0, zorder=2)
        ax2.plot(r["calibrated_log_loss"], yv, "o", markersize=9, color=col,
                 markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=3)
        ax2.annotate(f"{r['calibrated_log_loss']:.5f}",
                     (r["calibrated_log_loss"], yv), textcoords="offset points",
                     xytext=(10, 0), va="center", fontsize=8.5, color=INK_SOFT)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(d["name"], fontsize=9)
    ax2.set_xlabel("log-loss μετά από calibration (χαμηλότερο = καλύτερο)",
                   color=INK_SOFT)
    ax2.set_title("Ίδιοι γύροι, ίδιο calibration", color=INK, fontsize=12, pad=12)
    # explicit limits + a plain formatter: the default ScalarFormatter factors
    # out a ~1.0 offset and prints it on top of the axis label
    ax2.set_xlim(lo - 0.10 * span, hi + 0.55 * span)
    ax2.set_ylim(-0.6, len(d) - 0.4)
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.5f}"))
    _style_axes(ax2)
    for t in ax2.get_yticklabels():
        t.set_color(INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
