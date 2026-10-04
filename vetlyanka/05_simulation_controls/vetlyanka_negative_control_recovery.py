"""
vetlyanka_negative_control_recovery.py   (step 05 -- negative control; run after 02_fits)

NEGATIVE CONTROL: synthetic data from a TRUE SINGLE-regime process (no second
transmission phase), then BOTH models are fitted with the full fitting
procedure. It checks that the method does not invent a second regime when none
exists -- the complement to recovery_simulation_study.py (positive control,
where a second regime IS present). Same question as the parametric bootstrap,
but with the full multi-start pipeline on a few datasets instead of a quick
one-start refit on many.

GROUND TRUTH = the real SINGLE-sigmoid fit, read from
02_fits/results/single/vetlyanka_polished_fit_single.csv (incl. its fixed T_perceive).

For each of NUM_REPLICATES synthetic datasets (Poisson noise on the weekly
increments of the true single-sigmoid curve), the SAME procedure as on the real
data and in the recovery study:
  1. Double-sigmoid: LHS scan + multi-start L-BFGS-B polish, x1 - x0 >= 1, k = 13.
  2. Single-sigmoid: T_perceive FIXED at the value just fitted by the double
     model on this synthetic dataset (how the real single fit borrows it),
     k = 10, LHS scan + multi-start polish.
  3. AIC / AICc / BIC, and diagnostics of double's fitted decline: x1 (is it at
     its upper bound?), the tail in weeks 20-22.

What to look for:
  * AIC/AICc/BIC should favour single (no second regime exists).
  * double's x1 pinned at its upper bound (20) or scattered across replicates
    means the "second regime" is not genuinely identified.

(Earlier versions: T_perceive was FREE in the single refit -- 11 parameters,
more flexible than the paper's model; the true parameters were typed in from
the legacy-alignment fit; the simulation stopped at week 21; and the search
was much smaller than the recovery study's -- 30,000 samples, 8 starts. All fixed here; the search
budget now matches recovery_simulation_study.py. Note: within the real bounds
(x1 <= 20) double must place its decline inside the observation window, so it
cannot fully mimic a single-regime curve and can fit WORSE than single on this
data -- the parametric bootstrap shows the same.)

Sampling: scipy.stats.qmc.LatinHypercube with fixed seeds; noise: numpy
default_rng(NOISE_SEED), a separate stream.

OBSERVATION TIMES: weeks 1..22 at t = 1..22 (same as the fits).

Writes: 05_simulation_controls/results/negative_control/
          vetlyanka_negative_control_results.csv
          vetlyanka_negative_control_summary.txt
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # .../vetlyanka/05_simulation_controls
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):  # walk up to vetlyanka/, which contains src/
    if os.path.dirname(ROOT) == ROOT:
        raise FileNotFoundError("Could not find the src/ folder above this script.")
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import qmc

from plague_double_sigmoid_model import plague_model as plague_model_double
from plague_single_sigmoid_model import plague_model as plague_model_single
from vetlyanka_bounds import PARAM_NAMES, BOUNDS, PARAM_NAMES_SINGLE, BOUNDS_SINGLE

# ---------------------------------------------------------------------
# Settings (search budget identical to recovery_simulation_study.py)
# ---------------------------------------------------------------------
LEGACY_ALIGNMENT = False
NUM_REPLICATES = 5
NUM_LHS_SAMPLES = 200000
TOP_K = 10
NOISE_SEED = 2026
LHS_SEED_BASE = 3000
X0_X1_MIN_GAP = 1.0        # must match polish_vetlyanka_fit_double.py
K_DOUBLE, K_SINGLE = 13, 10

TRUTH_PATH = os.path.join(ROOT, "02_fits", "results", "single", "vetlyanka_polished_fit_single.csv")
OUTPUT_DIR = os.path.join(HERE, "results", "negative_control")
os.makedirs(OUTPUT_DIR, exist_ok=True)

t_start, t_end, dt = 0.0, 22.0, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
n_weeks = 22
OBS = slice(0, n_weeks) if LEGACY_ALIGNMENT else slice(1, n_weeks + 1)
OBS_WEEKS = np.arange(0, n_weeks) if LEGACY_ALIGNMENT else np.arange(1, n_weeks + 1)
TAIL_WEEKS = [20, 21, 22]
TAIL_IDX = [int(np.where(OBS_WEEKS == w)[0][0]) for w in TAIL_WEEKS] if not LEGACY_ALIGNMENT else [19, 20, 21]
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]
X1_C1_PLACEHOLDER = 1.0

IDX_X0, IDX_X1 = PARAM_NAMES.index("x0"), PARAM_NAMES.index("x1")
IDX_C1, IDX_TP = PARAM_NAMES.index("c1"), PARAM_NAMES.index("T_perceive")
X1_UPPER = BOUNDS[IDX_X1][1]
SINGLE_FREE = [n for n in PARAM_NAMES_SINGLE if n != "T_perceive"]           # 10
SINGLE_FREE_BOUNDS = [b for n, b in zip(PARAM_NAMES_SINGLE, BOUNDS_SINGLE) if n != "T_perceive"]


# ---------------------------------------------------------------------
# Simulation and helpers (identical to recovery_simulation_study.py)
# ---------------------------------------------------------------------
def simulate_weekly(model_fn, full_params):
    try:
        sol = solve_ivp(model_fn, [t_start, t_end], initial_conditions,
                        args=(full_params,), t_eval=t_points, method="RK45")
    except (ValueError, FloatingPointError, OverflowError):
        return None
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    weekly = sol.y[5][::steps_per_week][OBS]
    return weekly if len(weekly) == n_weeks else None


def to_incremental(cumulative, baseline=0.0):
    return np.diff(np.concatenate(([baseline], np.asarray(cumulative, dtype=float))))


def information_criteria(observed, predicted, k):
    n = observed.size
    s = max(float(np.sum((observed - predicted) ** 2)), np.finfo(float).tiny)
    aic = n * np.log(s / n) + 2 * k
    bic = n * np.log(s / n) + k * np.log(n)
    aicc = np.inf if n <= k + 1 else aic + 2 * k * (k + 1) / (n - k - 1)
    return aic, aicc, bic


def lhs(bounds, n, seed):
    lo = np.array([b[0] for b in bounds], float); hi = np.array([b[1] for b in bounds], float)
    return lo + qmc.LatinHypercube(d=len(bounds), seed=seed).random(n=n) * (hi - lo)


def multistart_fit(objective, bounds, n_samples, seed, top_k):
    samples = lhs(bounds, n_samples, seed)
    scores = np.array([objective(row) for row in samples])
    best_sse, best_p = np.inf, None
    for i in np.argsort(scores)[:top_k]:
        start = samples[i]
        sf = np.maximum(np.abs(start), 1e-3)
        res = minimize(lambda p: objective(p * sf), start / sf, method="L-BFGS-B",
                       bounds=[(lo / s, hi / s) for (lo, hi), s in zip(bounds, sf)],
                       options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 100})
        p = res.x * sf
        v = objective(p)
        if v < best_sse:
            best_sse, best_p = v, p
    return best_p, best_sse


def sse_of(model_fn, full, target):
    pred = simulate_weekly(model_fn, full)
    if pred is None:
        return 1e12
    v = float(np.sum((target - pred) ** 2))
    return v if np.isfinite(v) else 1e12


def double_objective(target):
    def f(p):
        if p[IDX_X0] >= p[IDX_X1] - X0_X1_MIN_GAP:
            return 1e12
        return sse_of(plague_model_double, p, target)
    return f


def single_full_factory(t_perceive):
    def full(reduced):
        d = dict(zip(SINGLE_FREE, reduced)); d["T_perceive"] = t_perceive
        return np.array([d.get(n, X1_C1_PLACEHOLDER) for n in PARAM_NAMES], dtype=float)
    return full


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if not os.path.exists(TRUTH_PATH):
        raise FileNotFoundError(f"{TRUTH_PATH} not found -- run 02_fits/polish_vetlyanka_fit_single_FIXED.py first.")
    truth = pd.read_csv(TRUTH_PATH)
    truth = dict(zip(truth["Parameter"], truth["Best_Fit"].astype(float)))
    true_full = single_full_factory(truth["T_perceive"])(np.array([truth[n] for n in SINGLE_FREE]))
    true_cum = simulate_weekly(plague_model_single, true_full)
    true_incr = np.clip(to_incremental(true_cum), 0, None)
    noise_rng = np.random.default_rng(NOISE_SEED)

    print(f"Observation times: {'t = 0..21 (LEGACY)' if LEGACY_ALIGNMENT else 't = 1..22 (end of each week)'}")
    print(f"Ground truth: single-sigmoid fit ({os.path.relpath(TRUTH_PATH, ROOT)}), "
          f"T_perceive = {truth['T_perceive']:.6f} -- NO second regime")
    print(f"True weekly deaths, weeks {TAIL_WEEKS}: {np.round(true_incr[TAIL_IDX], 3)}")
    print(f"{NUM_REPLICATES} replicates, {NUM_LHS_SAMPLES} LHS samples and {TOP_K} polished starts per model\n")

    rows, t0 = [], time.time()
    for r in range(NUM_REPLICATES):
        print(f"{'=' * 60}\nReplicate {r + 1}/{NUM_REPLICATES}\n{'=' * 60}", flush=True)
        synth_incr = noise_rng.poisson(true_incr).astype(float)
        synth_cum = np.cumsum(synth_incr)

        print("  fitting double-sigmoid...", flush=True)
        d_par, d_sse = multistart_fit(double_objective(synth_cum), BOUNDS, NUM_LHS_SAMPLES,
                                      LHS_SEED_BASE + 2 * r, TOP_K)
        d_pred = simulate_weekly(plague_model_double, d_par)
        tp = d_par[IDX_TP]

        print(f"  fitting single-sigmoid (T_perceive fixed at {tp:.4f} from this double fit)...", flush=True)
        s_full = single_full_factory(tp)
        s_red, s_sse = multistart_fit(lambda p: sse_of(plague_model_single, s_full(p), synth_cum),
                                      SINGLE_FREE_BOUNDS, NUM_LHS_SAMPLES, LHS_SEED_BASE + 2 * r + 1, TOP_K)
        s_pred = simulate_weekly(plague_model_single, s_full(s_red))

        aic_d, aicc_d, bic_d = information_criteria(synth_cum, d_pred, K_DOUBLE)
        aic_s, aicc_s, bic_s = information_criteria(synth_cum, s_pred, K_SINGLE)
        x1 = d_par[IDX_X1]
        at_bound = (X1_UPPER - x1) < 0.05 * (BOUNDS[IDX_X1][1] - BOUNDS[IDX_X1][0])
        d_tail = to_incremental(d_pred)[TAIL_IDX]; s_tail = to_incremental(s_pred)[TAIL_IDX]

        print(f"  single: SSE={s_sse:.2f}  AIC={aic_s:.2f}  AICc={aicc_s:.2f}  BIC={bic_s:.2f}")
        print(f"  double: SSE={d_sse:.2f}  AIC={aic_d:.2f}  AICc={aicc_d:.2f}  BIC={bic_d:.2f}  "
              f"x1={x1:.2f}{' (AT UPPER BOUND)' if at_bound else ''}")
        print(f"  favoured -- AIC: {'single' if aic_s < aic_d else 'DOUBLE'}, "
              f"AICc: {'single' if aicc_s < aicc_d else 'DOUBLE'}, BIC: {'single' if bic_s < bic_d else 'DOUBLE'}"
              f"   [{(time.time() - t0) / 60:.0f} min elapsed]")

        row = {"replicate": r + 1, "sse_single": s_sse, "sse_double": d_sse,
               "single_T_perceive_fixed": tp,
               "aic_single": aic_s, "aic_double": aic_d, "aicc_single": aicc_s, "aicc_double": aicc_d,
               "bic_single": bic_s, "bic_double": bic_d,
               "delta_AIC": aic_s - aic_d, "delta_AICc": aicc_s - aicc_d, "delta_BIC": bic_s - bic_d,
               "x1_fitted": x1, "x1_at_upper_bound": bool(at_bound), "c1_fitted": d_par[IDX_C1]}
        for w, dv, sv, yv in zip(TAIL_WEEKS, d_tail, s_tail, synth_incr[TAIL_IDX]):
            row[f"synthetic_wk{w}"] = yv; row[f"double_tail_wk{w}"] = dv; row[f"single_tail_wk{w}"] = sv
        for name, v in zip(PARAM_NAMES, d_par):
            row[f"double_{name}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "vetlyanka_negative_control_results.csv"), index=False)
    n = len(df)
    lines = [
        f"Replicates: {n}  (truth: single-sigmoid, NO second regime)",
        f"AIC favours single in {(df.delta_AIC < 0).sum()}/{n}  (delta AIC single-double: "
        f"{', '.join(f'{v:+.2f}' for v in df.delta_AIC)})",
        f"AICc favours single in {(df.delta_AICc < 0).sum()}/{n}",
        f"BIC favours single in {(df.delta_BIC < 0).sum()}/{n}",
        f"Raw SSE: double <= single in {(df.sse_double <= df.sse_single).sum()}/{n} "
        f"(within the real bounds x1 <= 20 double must decline inside the window, so it cannot fully "
        f"mimic a single-regime curve and may fit worse even with a thorough search)",
        f"double x1: {', '.join(f'{v:.2f}' for v in df.x1_fitted)}  "
        f"(mean {df.x1_fitted.mean():.2f}, sd {df.x1_fitted.std():.2f}; at upper bound in "
        f"{int(df.x1_at_upper_bound.sum())}/{n})",
    ]
    summary = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "vetlyanka_negative_control_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}\n{summary}\n\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()