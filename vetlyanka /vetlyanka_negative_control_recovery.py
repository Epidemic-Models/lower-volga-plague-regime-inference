"""
vetlyanka_negative_control_recovery.py

The missing negative control: generates synthetic data from a TRUE
SINGLE-regime process (no second transmission phase at all), then fits
BOTH the single- and double-regime models to it, using the same
multi-start procedure used throughout. This directly tests whether the
method correctly avoids inventing a spurious second regime when none
exists -- the complement to the existing recovery-simulation study,
which tested recoverability when a TRUE double-regime process IS
present.

What to look for:
- Does AIC/AICc/BIC correctly favor single-regime despite double's
  near-inevitable lower raw SSE (double always has more free parameters
  to chase noise with)?
- Does double's fitted x1 look genuinely identified (sharp, stable,
  consistent across replicates) or spurious (scattered across
  replicates, pinned at a bound, or associated with a near-zero decline
  amplitude, i.e. c1~0 or b2~b1 collapsing S_dec(t)~1)?

*** VERIFY BEFORE RUNNING ***
TRUE_SINGLE_PARAMS below are PLACEHOLDER values approximating Vetlyanka's
real confirmed single-sigmoid fit. Given two earlier scripts this
session silently used wrong reconstructed values (bounds, then observed
data) and produced misleading results before being caught, do NOT trust
these numbers as-is -- replace them with the real, confirmed
single-sigmoid best-fit parameters from your own polished_fit.csv
before running for real.

Required:
    plague_single_sigmoid_model.py, plague_double_sigmoid_model.py
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import qmc

import os
import signal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from plague_single_sigmoid_model import plague_model as plague_model_single
from plague_double_sigmoid_model import plague_model as plague_model_double

# Confirmed, real single-sigmoid fit (SSE=960.586, matches the established
# five-model table value of 960.585 -- T_perceive fixed at 3.571, not searched,
# per the established methodology for this model)
TRUE_SINGLE_PARAMS = {
    "b1": 0.19999999999999998, "b2": 5.17392745353192, "x0": 7.345928253581863,
    "c": 1.6510109445171477, "gamma1": 0.3499999999999999, "gamma2": 1.4,
    "mu1": 1.4, "mu2": 1.75, "T_perceive": 3.571,
    "sensitivity": 0.016906371566941407, "dispose_rate": 7.5,
}

N_REPLICATES = 5
NOISE_SD_FRACTION = 0.05  # matches the noise level already used in the existing recovery study -- verify
RANDOM_SEED = 2026

# Real, authoritative Vetlyanka bounds (from vetlyanka_bounds.py)
BOUNDS_SINGLE = [
    (0.001, 0.20), (5.00, 25.00), (5, 20), (0.02, 2.0),
    (0.35, 0.7), (1.167, 1.4), (1.167, 1.4), (1.75, 14),
    (0.5, 20.0), (0.005, 0.2), (2.0, 7.5),
]
PARAM_NAMES_SINGLE = ["b1", "b2", "x0", "c", "gamma1", "gamma2", "mu1", "mu2",
                      "T_perceive", "sensitivity", "dispose_rate"]

BOUNDS_DOUBLE = [
    (0.001, 0.20), (5.00, 25.00), (5, 20), (14, 20), (0.02, 2.0), (0.8, 4.0),
    (0.35, 0.7), (1.167, 1.4), (1.167, 1.4), (1.75, 14),
    (0.5, 20.0), (0.005, 0.2), (2.0, 7.5),
]
PARAM_NAMES_DOUBLE = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2",
                      "mu1", "mu2", "T_perceive", "sensitivity", "dispose_rate"]
IDX_X0_D, IDX_X1_D = 2, 3
X0_X1_MIN_GAP = 1.0

n_weeks = 22
t_start, t_end, dt = 0, n_weeks - 1, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]

NUM_LHS_SAMPLES = 30000
TOP_K = 8


class TimeoutError_(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError_()


def simulate_weekly_single(params_dict):
    p = np.array([params_dict[n] for n in PARAM_NAMES_SINGLE])
    x1_c1_placeholder = 1.0
    full = np.array([p[0], p[1], p[2], x1_c1_placeholder, p[3], x1_c1_placeholder,
                      p[4], p[5], p[6], p[7], p[8], p[9], p[10]])
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(5)
    try:
        sol = solve_ivp(plague_model_single, [t_start, t_end], initial_conditions,
                         args=(full,), t_eval=t_points, method="RK45",
                         rtol=1e-6, atol=1e-8)
    except (ValueError, FloatingPointError, OverflowError, TimeoutError_):
        return None
    finally:
        signal.alarm(0)
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    return sol.y[5][::steps_per_week][:n_weeks]  # DR, all-cause -- correct for Vetlyanka


def generate_synthetic_replicates():
    """Poisson noise on WEEKLY incremental deaths (matching the existing
    positive-control recovery study's methodology exactly), not Gaussian
    noise on the cumulative curve -- weekly deaths are genuine count data,
    and Poisson is the statistically appropriate model, particularly at
    the low counts common in the early weeks of this outbreak."""
    true_cumulative = simulate_weekly_single(TRUE_SINGLE_PARAMS)
    if true_cumulative is None:
        raise RuntimeError("True-parameter simulation failed -- check TRUE_SINGLE_PARAMS.")
    true_weekly = np.diff(true_cumulative, prepend=0)
    true_weekly = np.maximum(true_weekly, 0)  # guard against solver noise producing tiny negatives
    rng = np.random.default_rng(RANDOM_SEED)
    replicates = []
    for i in range(N_REPLICATES):
        noisy_weekly = rng.poisson(true_weekly)
        noisy_cumulative = np.cumsum(noisy_weekly).astype(float)
        replicates.append(noisy_cumulative)
    return true_cumulative, replicates


def scale_parameters(params):
    sf = np.maximum(np.abs(params), 1e-3)
    return params / sf, sf


def fit_single(observed):
    def sse(reduced):
        p = np.array([reduced[i] for i in range(len(PARAM_NAMES_SINGLE))])
        d = dict(zip(PARAM_NAMES_SINGLE, p))
        pred = simulate_weekly_single(d)
        if pred is None:
            return 1e12
        s = np.sum((observed - pred) ** 2)
        return float(s) if np.isfinite(s) else 1e12

    sampler = qmc.LatinHypercube(d=len(BOUNDS_SINGLE), seed=42)
    unit = sampler.random(n=NUM_LHS_SAMPLES)
    samples = np.array([[row[j] * (BOUNDS_SINGLE[j][1] - BOUNDS_SINGLE[j][0]) + BOUNDS_SINGLE[j][0]
                          for j in range(len(BOUNDS_SINGLE))] for row in unit])
    scored = [(sse(row), row) for row in samples]
    scored.sort(key=lambda p: p[0])
    top = scored[:TOP_K]
    best_sse, best_params = np.inf, None
    for lhs_sse, start in top:
        sx0, sf = scale_parameters(start)
        sb = [(lo/f, hi/f) for (lo, hi), f in zip(BOUNDS_SINGLE, sf)]
        result = minimize(lambda sp: sse(sp * sf), sx0, method="L-BFGS-B", bounds=sb,
                           options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})
        opt = result.x * sf
        psse = sse(opt)
        if psse < best_sse:
            best_sse, best_params = psse, opt
    return best_sse, best_params


def fit_double(observed):
    def sse(params):
        if params[IDX_X0_D] >= params[IDX_X1_D] - X0_X1_MIN_GAP:
            return 1e12
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(5)
        try:
            sol = solve_ivp(plague_model_double, [t_start, t_end], initial_conditions,
                             args=(params,), t_eval=t_points, method="RK45",
                             rtol=1e-6, atol=1e-8)
        except (ValueError, FloatingPointError, OverflowError, TimeoutError_):
            return 1e12
        finally:
            signal.alarm(0)
        if not sol.success or not np.all(np.isfinite(sol.y)):
            return 1e12
        pred = sol.y[5][::steps_per_week][:n_weeks]
        s = np.sum((observed - pred) ** 2)
        return float(s) if np.isfinite(s) else 1e12

    sampler = qmc.LatinHypercube(d=len(BOUNDS_DOUBLE), seed=42)
    unit = sampler.random(n=NUM_LHS_SAMPLES)
    samples = np.array([[row[j] * (BOUNDS_DOUBLE[j][1] - BOUNDS_DOUBLE[j][0]) + BOUNDS_DOUBLE[j][0]
                          for j in range(len(BOUNDS_DOUBLE))] for row in unit])
    scored = [(sse(row), row) for row in samples]
    scored.sort(key=lambda p: p[0])
    top = scored[:TOP_K]
    best_sse, best_params = np.inf, None
    for lhs_sse, start in top:
        sx0, sf = scale_parameters(start)
        sb = [(lo/f, hi/f) for (lo, hi), f in zip(BOUNDS_DOUBLE, sf)]
        result = minimize(lambda sp: sse(sp * sf), sx0, method="L-BFGS-B", bounds=sb,
                           options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})
        opt = result.x * sf
        psse = sse(opt)
        if psse < best_sse:
            best_sse, best_params = psse, opt
    return best_sse, best_params


def aic(sse, k, n):
    return n * np.log(sse / n) + 2 * k


def bic(sse, k, n):
    return n * np.log(sse / n) + k * np.log(n)


def aicc(sse, k, n):
    a = aic(sse, k, n)
    return a + (2*k*(k+1))/(n-k-1) if n > k+1 else np.inf


def main():
    print("Generating synthetic data from a TRUE single-regime process...")
    true_curve, replicates = generate_synthetic_replicates()

    results = []
    for i, obs in enumerate(replicates):
        print(f"\n--- Replicate {i+1}/{N_REPLICATES} ---")
        sse_s, params_s = fit_single(obs)
        sse_d, params_d = fit_double(obs)
        x1_fitted = params_d[PARAM_NAMES_DOUBLE.index("x1")]
        n = n_weeks
        aic_s, aic_d = aic(sse_s, 11, n), aic(sse_d, 13, n)
        aicc_s, aicc_d = aicc(sse_s, 11, n), aicc(sse_d, 13, n)
        bic_s, bic_d = bic(sse_s, 11, n), bic(sse_d, 13, n)
        print(f"  single: SSE={sse_s:.2f}, AIC={aic_s:.2f}, AICc={aicc_s:.2f}, BIC={bic_s:.2f}")
        print(f"  double: SSE={sse_d:.2f}, AIC={aic_d:.2f}, AICc={aicc_d:.2f}, BIC={bic_d:.2f}, x1={x1_fitted:.2f}")
        print(f"  AIC favors: {'single' if aic_s < aic_d else 'double'}  "
              f"AICc favors: {'single' if aicc_s < aicc_d else 'double'}  "
              f"BIC favors: {'single' if bic_s < bic_d else 'double'}")
        results.append({"replicate": i+1, "sse_single": sse_s, "sse_double": sse_d,
                         "aic_single": aic_s, "aic_double": aic_d,
                         "aicc_single": aicc_s, "aicc_double": aicc_d,
                         "bic_single": bic_s, "bic_double": bic_d,
                         "x1_fitted": x1_fitted})

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "vetlyanka_negative_control_results.csv"), index=False)
    print(f"\n{'='*60}\nSUMMARY across {N_REPLICATES} replicates\n{'='*60}")
    print(df[["replicate", "sse_single", "sse_double", "x1_fitted"]])
    print(f"\nx1 fitted across replicates (true model has NO real x1 -- look for scatter/instability):")
    print(f"  mean={df['x1_fitted'].mean():.2f}, std={df['x1_fitted'].std():.2f}, "
          f"range=[{df['x1_fitted'].min():.2f}, {df['x1_fitted'].max():.2f}]")
    print(f"\nAIC favors single in {(df['aic_single'] < df['aic_double']).sum()}/{N_REPLICATES} replicates")
    print(f"AICc favors single in {(df['aicc_single'] < df['aicc_double']).sum()}/{N_REPLICATES} replicates")
    print(f"BIC favors single in {(df['bic_single'] < df['bic_double']).sum()}/{N_REPLICATES} replicates")
    print(f"\nSaved {OUTPUT_DIR}/vetlyanka_negative_control_results.csv")


if __name__ == "__main__":
    main()