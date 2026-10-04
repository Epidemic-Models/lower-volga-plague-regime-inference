"""
recovery_simulation_study.py   (step 05 -- positive control; run after 02_fits)

Parameter-recovery / calibration study (Reviewer #3: "fitting the models on
simulated data would clearly strengthen the manuscript"; Reviewer #1: do the
better fits "simply emerge because there are additional parameters").

GROUND TRUTH = the real DOUBLE-sigmoid fit (two regimes genuinely exist).
QUESTION: when two regimes really exist, does the actual fitting procedure
recover the true parameters (especially the shift time x1), and does the
single-sigmoid fail in the same way it does on the real data (persistent
tail instead of a halt)?

For each of NUM_REPLICATES synthetic datasets (Poisson noise on the weekly
increments of the true double-sigmoid curve), the script runs the SAME
procedure as on the real data:
  1. Double-sigmoid: LHS scan + multi-start L-BFGS-B polish, x1 - x0 >= 1.
  2. Single-sigmoid: T_perceive FIXED at the value just fitted by the double
     model on this same synthetic dataset (exactly how the real single fit
     borrows T_perceive from the real double fit), 10 free parameters,
     LHS scan + multi-start polish.
  3. Tail check: weekly deaths in weeks 20-22 (the weeks with zero deaths in
     the real data) for both refits.
  4. AIC / AICc / BIC with k = 13 (double) and k = 10 (single).
  5. Recovery error of every double-sigmoid parameter, in particular x1.

(An earlier version left T_perceive FREE in the single refit -- 11 parameters,
more flexible than the model actually used in the paper. That is fixed here.)

Sampling: scipy.stats.qmc.LatinHypercube with fixed seeds (reproducible).
Noise: numpy default_rng(NOISE_SEED). The two use separate random streams, so
changing NUM_LHS_SAMPLES does not change the synthetic datasets.

OBSERVATION TIMES: weeks 1..22 at t = 1..22 (same as the fits).

Reads:  02_fits/results/double/vetlyanka_polished_fit.csv   (ground truth)
Writes: 05_simulation_controls/results/recovery/
          recovery_study_results.csv   -- one row per replicate
          recovery_study_summary.txt
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
# Settings
# ---------------------------------------------------------------------
LEGACY_ALIGNMENT = False
NUM_REPLICATES = 5
NUM_LHS_SAMPLES = 200000   # per model per replicate (real fits used 400,000)
TOP_K = 10                 # polished starts per model per replicate (real fits used 20)
NOISE_SEED = 0
LHS_SEED_BASE = 1000       # replicate r uses seeds LHS_SEED_BASE + 2r (double), + 2r + 1 (single)
X0_X1_MIN_GAP = 1.0        # must match polish_vetlyanka_fit_double.py
K_DOUBLE, K_SINGLE = 13, 10

TRUTH_PATH = os.path.join(ROOT, "02_fits", "results", "double", "vetlyanka_polished_fit.csv")
OUTPUT_DIR = os.path.join(HERE, "results", "recovery")
os.makedirs(OUTPUT_DIR, exist_ok=True)

t_start, t_end, dt = 0.0, 22.0, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
n_weeks = 22
OBS = slice(0, n_weeks) if LEGACY_ALIGNMENT else slice(1, n_weeks + 1)
OBS_WEEKS = np.arange(0, n_weeks) if LEGACY_ALIGNMENT else np.arange(1, n_weeks + 1)
TAIL_WEEKS = [20, 21, 22]                       # real data: 0 deaths in these weeks
TAIL_IDX = [int(np.where(OBS_WEEKS == w)[0][0]) for w in TAIL_WEEKS] if not LEGACY_ALIGNMENT else [19, 20, 21]
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]
X1_C1_PLACEHOLDER = 1.0

IDX_X0, IDX_X1 = PARAM_NAMES.index("x0"), PARAM_NAMES.index("x1")
IDX_TP = PARAM_NAMES.index("T_perceive")
SINGLE_FREE = [n for n in PARAM_NAMES_SINGLE if n != "T_perceive"]           # 10
SINGLE_FREE_BOUNDS = [b for n, b in zip(PARAM_NAMES_SINGLE, BOUNDS_SINGLE) if n != "T_perceive"]


# ---------------------------------------------------------------------
# Simulation and helpers
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
    """LHS scan + TOP_K L-BFGS-B polishes (same settings as the 02_fits scripts)."""
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
        raise FileNotFoundError(f"{TRUTH_PATH} not found -- run 02_fits/polish_vetlyanka_fit_double.py first.")
    truth_df = pd.read_csv(TRUTH_PATH).set_index("Parameter")
    true_params = np.array([truth_df.loc[n, "Best_Fit"] for n in PARAM_NAMES], dtype=float)
    true_cum = simulate_weekly(plague_model_double, true_params)
    true_incr = np.clip(to_incremental(true_cum), 0, None)
    noise_rng = np.random.default_rng(NOISE_SEED)

    print(f"Observation times: {'t = 0..21 (LEGACY)' if LEGACY_ALIGNMENT else 't = 1..22 (end of each week)'}")
    print(f"Ground truth: double-sigmoid fit ({os.path.relpath(TRUTH_PATH, ROOT)}), true x1 = {true_params[IDX_X1]:.3f}")
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
        d_tail = to_incremental(d_pred)[TAIL_IDX]
        s_tail = to_incremental(s_pred)[TAIL_IDX]
        x1_err = 100 * (d_par[IDX_X1] - true_params[IDX_X1]) / true_params[IDX_X1]

        print(f"  synthetic tail {TAIL_WEEKS}: {np.round(synth_incr[TAIL_IDX], 2)}")
        print(f"  double tail:  {np.round(d_tail, 2)}  SSE={d_sse:.2f}   x1={d_par[IDX_X1]:.3f} ({x1_err:+.2f}%)")
        print(f"  single tail:  {np.round(s_tail, 2)}  SSE={s_sse:.2f}")
        print(f"  dAIC={aic_s - aic_d:.2f}  dAICc={aicc_s - aicc_d:.2f}  dBIC={bic_s - bic_d:.2f}  "
              f"(single - double; positive favours double)   [{(time.time() - t0) / 60:.0f} min elapsed]")

        row = {"replicate": r + 1, "double_sse": d_sse, "single_sse": s_sse,
               "single_T_perceive_fixed": tp,
               "delta_AIC": aic_s - aic_d, "delta_AICc": aicc_s - aicc_d, "delta_BIC": bic_s - bic_d,
               "x1_error_pct": x1_err}
        for w, dv, sv, yv in zip(TAIL_WEEKS, d_tail, s_tail, synth_incr[TAIL_IDX]):
            row[f"synthetic_wk{w}"] = yv; row[f"double_tail_wk{w}"] = dv; row[f"single_tail_wk{w}"] = sv
        for name, tv, rv in zip(PARAM_NAMES, true_params, d_par):
            row[f"true_{name}"] = tv; row[f"recovered_{name}"] = rv
            row[f"error_pct_{name}"] = 100 * (rv - tv) / tv if tv != 0 else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "recovery_study_results.csv"), index=False)

    dtail = df[[f"double_tail_wk{w}" for w in TAIL_WEEKS]].to_numpy()
    stail = df[[f"single_tail_wk{w}" for w in TAIL_WEEKS]].to_numpy()
    lines = [
        f"Replicates: {len(df)}",
        f"AIC favours double in {(df.delta_AIC > 0).sum()}/{len(df)} "
        f"(range {df.delta_AIC.min():.2f} to {df.delta_AIC.max():.2f})",
        f"BIC favours double in {(df.delta_BIC > 0).sum()}/{len(df)}; "
        f"AICc favours double in {(df.delta_AICc > 0).sum()}/{len(df)}",
        f"x1 recovery error (%): {', '.join(f'{v:+.2f}' for v in df.x1_error_pct)} "
        f"(max |error| {df.x1_error_pct.abs().max():.2f}%)",
        f"Mean weekly deaths in weeks {TAIL_WEEKS} -- double refit: {dtail.mean():.3f}, single refit: {stail.mean():.3f}",
        f"Single refit tail > double refit tail in {(stail.mean(axis=1) > dtail.mean(axis=1)).sum()}/{len(df)} replicates",
        "Median |recovery error| per parameter (%): " + ", ".join(
            f"{n}={df[f'error_pct_{n}'].abs().median():.1f}" for n in PARAM_NAMES),
    ]
    summary = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "recovery_study_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}\n{summary}\n\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()