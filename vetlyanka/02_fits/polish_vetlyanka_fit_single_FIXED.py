"""
polish_vetlyanka_fit_single_FIXED.py   (step 02 -- run AFTER polish_vetlyanka_fit_double.py)

Fit of the Vetlyanka SIGMOIDAL single-sigmoid model: rise-only beta(t),
behavioural feedback, gamma1/gamma2/mu1/mu2 fitted (time-varying, NOT frozen).

Free parameters (k = 10):
    b1, b2, x0, c, gamma1, gamma2, mu1, mu2, sensitivity, dispose_rate
x1 and c1 do not exist in this model. T_perceive is FIXED (not searched).

WHY T_perceive IS FIXED: when left free, T_perceive runs to whatever ceiling
it is given while dispose_rate drops to its floor -- a joint escape route
(a very long perception lag plus the slowest corpse disposal together fake
a suppression effect). So T_perceive is fixed at the value fitted by the
DOUBLE-sigmoid model of the same gamma/mu family, where it converges
cleanly. That value is READ AUTOMATICALLY from
    02_fits/results/double/vetlyanka_polished_fit.csv
so this script must run after polish_vetlyanka_fit_double.py.
(Published version: 3.571, legacy alignment.)

Reads 01_sampling/parameter_samples_<NUM_SAMPLES>.csv and writes to
02_fits/results/single/.

OBSERVATION TIMES: end-of-week cumulative deaths for weeks 1..22, compared
at t = 1..22. LEGACY_ALIGNMENT = True reproduces the published t = 0..21.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # .../vetlyanka/02_fits
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):  # walk up to vetlyanka/, which contains src/
    if os.path.dirname(ROOT) == ROOT:
        raise FileNotFoundError("Could not find the src/ folder above this script.")
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from plague_single_sigmoid_model import plague_model
from vetlyanka_bounds import PARAM_NAMES, PARAM_NAMES_SINGLE, BOUNDS_SINGLE

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
LEGACY_ALIGNMENT = False   # True = published (t = 0..21); False = corrected (t = 1..22)

SAMPLES_DIR = os.path.join(ROOT, "01_sampling")
OUTPUT_DIR = os.path.join(HERE, "results", "single")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 400000  # must match 01_sampling/parameter_samples_<NUM_SAMPLES>.csv
SAMPLES_PATH = os.path.join(SAMPLES_DIR, f"parameter_samples_{NUM_SAMPLES}.csv")
TOP_K = 20            # polish this many independent starting points, keep the best

# T_perceive is taken from the double-sigmoid fit (same gamma/mu family).
DOUBLE_FIT_PATH = os.path.join(HERE, "results", "double", "vetlyanka_polished_fit.csv")
T_PERCEIVE_OVERRIDE = None  # set a number (e.g. 3.571) to force a value instead of reading the file

FIT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "vetlyanka_polished_fit_single.csv")
PREDICTIONS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "vetlyanka_single_predictions.csv")
SUMMARY_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "vetlyanka_single_summary.txt")
MULTISTART_LOG_PATH = os.path.join(OUTPUT_DIR, "vetlyanka_single_FIXED_multistart_log.csv")
X1_C1_PLACEHOLDER = 1.0  # x1, c1 are not used by the single-sigmoid model

t_start, t_end, dt = 0.0, 22.0, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
n_weeks = 22
OBS = slice(0, n_weeks) if LEGACY_ALIGNMENT else slice(1, n_weeks + 1)
OBS_WEEKS = np.arange(0, n_weeks) if LEGACY_ALIGNMENT else np.arange(1, n_weeks + 1)
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]  # S0, I0, R0, DI0, DDI0, DR0, P0


def load_t_perceive():
    if T_PERCEIVE_OVERRIDE is not None:
        return float(T_PERCEIVE_OVERRIDE), "manual override"
    if not os.path.exists(DOUBLE_FIT_PATH):
        raise FileNotFoundError(f"{DOUBLE_FIT_PATH} not found -- run polish_vetlyanka_fit_double.py first.")
    fit = pd.read_csv(DOUBLE_FIT_PATH)
    value = float(fit.loc[fit["Parameter"] == "T_perceive", "Best_Fit"].iloc[0])
    return value, os.path.relpath(DOUBLE_FIT_PATH, ROOT)


T_PERCEIVE_FIXED, T_PERCEIVE_SOURCE = load_t_perceive()

# Columns of the shared 13-column sample file used here (no x1, c1, T_perceive)
_FULL_TO_SINGLE_IDX = [i for i, name in enumerate(PARAM_NAMES)
                       if name in PARAM_NAMES_SINGLE and name != "T_perceive"]
REDUCED_PARAM_NAMES = [name for name in PARAM_NAMES_SINGLE if name != "T_perceive"]
REDUCED_BOUNDS = [b for name, b in zip(PARAM_NAMES_SINGLE, BOUNDS_SINGLE) if name != "T_perceive"]
K_REDUCED = len(REDUCED_PARAM_NAMES)  # 10
if K_REDUCED != 10:
    raise ValueError(f"Expected 10 free parameters, got {K_REDUCED}: {REDUCED_PARAM_NAMES}")


def load_observed_data():
    """Vetlyanka cumulative deaths at the end of weeks 1..22."""
    return np.array([3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19, 27, 34,
                     90, 259, 313, 345, 364, 376, 376, 376, 376], dtype=float)


def expand_reduced_to_full(reduced_params):
    d = dict(zip(REDUCED_PARAM_NAMES, reduced_params))
    d["T_perceive"] = T_PERCEIVE_FIXED
    return np.array([d.get(name, X1_C1_PLACEHOLDER) for name in PARAM_NAMES], dtype=float)


# ---------------------------------------------------------------------
# Simulation and objective
# ---------------------------------------------------------------------
def simulate_cumulative(reduced_params, t_eval=None):
    try:
        sol = solve_ivp(plague_model, [t_start, t_end], initial_conditions,
                        args=(expand_reduced_to_full(reduced_params),),
                        t_eval=t_points, method="RK45")
    except (ValueError, FloatingPointError, OverflowError):
        return None
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    predicted = sol.y[5][::steps_per_week][OBS]  # DR = cumulative deaths at the observation times
    return predicted if predicted.shape[0] == n_weeks else None


def sse_unscaled(reduced_params, observed):
    predicted = simulate_cumulative(reduced_params)
    if predicted is None:
        return 1e12
    sse = float(np.sum((observed - predicted) ** 2))
    return sse if np.isfinite(sse) else 1e12


def sse_objective(scaled_params, scaling_factors, observed):
    return sse_unscaled(scaled_params * scaling_factors, observed)


def scale_parameters(params):
    scaling_factors = np.maximum(np.abs(params), 1e-3)
    return params / scaling_factors, scaling_factors


def information_criteria(sse, n, k):
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    aicc = aic + 2 * k * (k + 1) / (n - k - 1)
    return aic, aicc, bic


def find_top_k_lhs_candidates(samples_path, observed, k=TOP_K):
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"{samples_path} not found -- run 01_sampling/parameter_sampling.py first.")
    full_samples = np.loadtxt(samples_path, delimiter=",", skiprows=1)
    if full_samples.ndim != 2 or full_samples.shape[1] != 13:
        raise ValueError(f"Expected 13 columns in {samples_path}.")
    print(f"Scanning {full_samples.shape[0]} LHS candidates...")
    scored = []
    for i, full_row in enumerate(full_samples):
        reduced_row = full_row[_FULL_TO_SINGLE_IDX]
        scored.append((sse_unscaled(reduced_row, observed), reduced_row))
        if (i + 1) % 20000 == 0:
            print(f"  scanned {i + 1}/{full_samples.shape[0]} "
                  f"(best so far: {min(s for s, _ in scored):.3f})", flush=True)
    scored.sort(key=lambda pair: pair[0])
    top_k = scored[:k]
    print(f"\nTop {k} LHS candidate SSEs: {[f'{s:.3f}' for s, _ in top_k]}")
    return top_k


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    observed = load_observed_data()
    print("Single-sigmoid (time-varying gamma, mu) Vetlyanka fit")
    print(f"Observation times: {'t = 0..21 (LEGACY)' if LEGACY_ALIGNMENT else 't = 1..22 (end of each week)'}")
    print(f"T_perceive FIXED at {T_PERCEIVE_FIXED:.6f}  (from {T_PERCEIVE_SOURCE})")
    print(f"Reading samples from: {SAMPLES_PATH}\nWriting outputs to:   {OUTPUT_DIR}\n")

    top_candidates = find_top_k_lhs_candidates(SAMPLES_PATH, observed)

    best_result, best_polished_sse, best_scaling_factors = None, np.inf, None
    multistart_log = []
    for rank, (lhs_sse, start_row) in enumerate(top_candidates):
        scaled_x0, scaling_factors = scale_parameters(start_row)
        scaled_bounds = [(lo / sf, hi / sf) for (lo, hi), sf in zip(REDUCED_BOUNDS, scaling_factors)]
        print(f"\nPolishing start #{rank + 1}/{len(top_candidates)} (LHS SSE={lhs_sse:.3f})...", flush=True)
        result = minimize(sse_objective, scaled_x0, args=(scaling_factors, observed),
                          method="L-BFGS-B", bounds=scaled_bounds,
                          options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 100})
        p = result.x * scaling_factors
        polished_sse = sse_unscaled(p, observed)
        print(f"  -> polished SSE={polished_sse:.6f}  (iters={result.nit}, evals={result.nfev})")
        multistart_log.append({"rank": rank + 1, "lhs_sse": lhs_sse, "polished_sse": polished_sse,
                               "iterations": result.nit, "evaluations": result.nfev})
        if polished_sse < best_polished_sse:
            best_polished_sse, best_result, best_scaling_factors = polished_sse, result, scaling_factors

    pd.DataFrame(multistart_log).to_csv(MULTISTART_LOG_PATH, index=False)
    best_rank = min(range(len(multistart_log)), key=lambda i: multistart_log[i]["polished_sse"]) + 1
    print(f"\nBest overall: rank #{best_rank} start, polished SSE = {best_polished_sse:.6f}")
    sorted_sse = sorted(r["polished_sse"] for r in multistart_log)
    n_agree = sum(s < sorted_sse[0] * 1.001 for s in sorted_sse)
    print(f"{n_agree} of {len(sorted_sse)} starts reached the best SSE (within 0.1%) -- "
          f"if only 1 did, confirm with a second run (NUM_SAMPLES = 200000).")

    optimized = best_result.x * best_scaling_factors
    polished_sse = sse_unscaled(optimized, observed)
    predicted = simulate_cumulative(optimized)
    aic, aicc, bic = information_criteria(polished_sse, n_weeks, K_REDUCED)

    # Fitted parameters, plus T_perceive as a FIXED row so later scripts can read it from here.
    fit_df = pd.DataFrame({
        "Parameter": REDUCED_PARAM_NAMES + ["T_perceive"],
        "Bound_lo": [b[0] for b in REDUCED_BOUNDS] + [np.nan],
        "Bound_hi": [b[1] for b in REDUCED_BOUNDS] + [np.nan],
        "Best_Fit": list(optimized) + [T_PERCEIVE_FIXED],
        "Status": ["fitted"] * K_REDUCED + ["fixed"],
    })
    fit_df.to_csv(FIT_OUTPUT_PATH, index=False)
    pd.DataFrame({"Week": OBS_WEEKS, "Observed_cumulative_deaths": observed,
                  "Predicted_cumulative_deaths": predicted,
                  "Residual": observed - predicted}).to_csv(PREDICTIONS_OUTPUT_PATH, index=False)

    near_bound = [name for name, (lo, hi), val in zip(REDUCED_PARAM_NAMES, REDUCED_BOUNDS, optimized)
                  if (val - lo) < 0.05 * (hi - lo) or (hi - val) < 0.05 * (hi - lo)]
    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(f"SSE={polished_sse:.6f}\nk={K_REDUCED}\nAIC={aic:.4f}\nAICc={aicc:.4f}\nBIC={bic:.4f}\n"
                 f"T_perceive_fixed={T_PERCEIVE_FIXED:.6f} (from {T_PERCEIVE_SOURCE})\n"
                 f"alignment={'legacy t=0..21' if LEGACY_ALIGNMENT else 't=1..22'}\n"
                 f"samples={os.path.basename(SAMPLES_PATH)}\nTOP_K={TOP_K}\n"
                 f"starts reaching best SSE: {n_agree}/{len(sorted_sse)}\n"
                 f"within 5% of a bound: {near_bound}\n")

    print("\n", fit_df.to_string(index=False))
    print(f"\nSSE={polished_sse:.6f}  AIC={aic:.3f}  AICc={aicc:.3f}  BIC={bic:.3f}  "
          f"(k={K_REDUCED}; T_perceive fixed, not counted)")
    print(f"\nWithin 5% of a bound: {near_bound or 'none'}")
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()