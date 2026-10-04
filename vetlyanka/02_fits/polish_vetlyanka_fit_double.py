"""
polish_vetlyanka_fit_double.py   (step 02 -- run this FIRST of the five fits)

Fit for the Vetlyanka double-sigmoid model.

Reads 01_sampling/parameter_samples_<NUM_SAMPLES>.csv (from
parameter_sampling.py) and writes to 02_fits/results/double/.
The single-sigmoid fit reads this model's fitted T_perceive from that
output, so this script must run before polish_vetlyanka_fit_single_FIXED.py.

PROCEDURE: score every LHS sample (SSE against the observed cumulative
deaths), keep the TOP_K best, polish each with L-BFGS-B (scaled
parameters, box bounds), and keep the lowest polished SSE.

OBSERVATION TIMES: the observed series is end-of-week cumulative deaths
for weeks 1..22, so the model is compared at t = 1, 2, ..., 22.
(The published version compared at t = 0..21, which forced model week 1
to 0 against an observed 3. Set LEGACY_ALIGNMENT = True to reproduce it.)

X0/X1 ORDERING CONSTRAINT: candidates with x1 - x0 < X0_X1_MIN_GAP are
rejected, so the rise and the decline stay two distinct transitions
(without it the optimiser can push x0 past x1 and let the decline term
do the curve-shaping, which breaks the two-regime interpretation).
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

from plague_double_sigmoid_model import plague_model
from vetlyanka_bounds import PARAMETERS, PARAM_NAMES, BOUNDS  # single shared source of truth

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
LEGACY_ALIGNMENT = False   # True = published (t = 0..21); False = corrected (t = 1..22)

SAMPLES_DIR = os.path.join(ROOT, "01_sampling")
OUTPUT_DIR = os.path.join(HERE, "results", "double")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 400000  # must match 01_sampling/parameter_samples_<NUM_SAMPLES>.csv # must match 01_sampling/parameter_samples_<NUM_SAMPLES>.csv
SAMPLES_PATH = os.path.join(SAMPLES_DIR, f"parameter_samples_{NUM_SAMPLES}.csv")
TOP_K = 20            # polish this many independent starting points, keep the best
MULTISTART_LOG_PATH = os.path.join(OUTPUT_DIR, "vetlyanka_double_multistart_log.csv")
X0_X1_MIN_GAP = 1.0   # require x1 - x0 >= this many weeks

t_start, t_end, dt = 0, 22, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
n_weeks = 22
OBS = slice(0, n_weeks) if LEGACY_ALIGNMENT else slice(1, n_weeks + 1)
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]  # S0, I0, R0, DI0, DDI0, DR0, P0

IDX_X0 = PARAM_NAMES.index("x0")
IDX_X1 = PARAM_NAMES.index("x1")


def scale_parameters(params):
    scaling_factors = np.maximum(np.abs(params), 1e-3)
    return params / scaling_factors, scaling_factors


def load_observed_data():
    """Vetlyanka cumulative deaths at the end of weeks 1..22."""
    return np.array([3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19, 27, 34,
                     90, 259, 313, 345, 364, 376, 376, 376, 376], dtype=float)


def simulate_cumulative(params):
    sol = solve_ivp(
        plague_model, [t_start, t_end], initial_conditions,
        args=(params,), t_eval=t_points, method="RK45",
    )
    if not sol.success:
        return None
    return sol.y[5][::steps_per_week][OBS]  # DR = cumulative deaths at the observation times


def sse_unscaled(params, observed):
    """Raw SSE, with the x0/x1 ordering constraint enforced first."""
    if params[IDX_X0] >= params[IDX_X1] - X0_X1_MIN_GAP:
        return 1e12
    predicted = simulate_cumulative(params)
    if predicted is None or predicted.shape[0] != observed.shape[0]:
        return 1e12
    sse = float(np.sum((observed - predicted) ** 2))
    return sse if np.isfinite(sse) else 1e12


def sse_objective(scaled_params, scaling_factors, observed):
    return sse_unscaled(scaled_params * scaling_factors, observed)


def find_top_k_lhs_candidates(samples_path, observed, k=TOP_K):
    """Score every LHS row and return the k rows with the lowest SSE."""
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"{samples_path} not found -- run 01_sampling/parameter_sampling.py first.")
    samples = np.loadtxt(samples_path, delimiter=",", skiprows=1)
    scored = []
    for i, row in enumerate(samples):
        scored.append((sse_unscaled(row, observed), row))
        if (i + 1) % 20000 == 0:
            print(f"  scanned {i + 1}/{samples.shape[0]} LHS candidates "
                  f"(best so far: {min(s for s, _ in scored):.3f})", flush=True)
    scored.sort(key=lambda pair: pair[0])
    top_k = scored[:k]
    print(f"\nTop {k} LHS candidate SSEs: {[f'{s:.3f}' for s, _ in top_k]}")
    return top_k


def main():
    observed = load_observed_data()
    print(f"Observation times: {'t = 0..21 (LEGACY)' if LEGACY_ALIGNMENT else 't = 1..22 (end of each week)'}")
    print(f"Scanning {SAMPLES_PATH} for the top {TOP_K} starting candidates "
          f"(x1 - x0 >= {X0_X1_MIN_GAP} weeks required)...")
    top_candidates = find_top_k_lhs_candidates(SAMPLES_PATH, observed)

    best_result, best_polished_sse, best_scaling_factors = None, np.inf, None
    multistart_log = []

    for rank, (lhs_sse, start_row) in enumerate(top_candidates):
        scaled_x0, scaling_factors = scale_parameters(start_row)
        scaled_bounds = [(lo / sf, hi / sf) for (lo, hi), sf in zip(BOUNDS, scaling_factors)]

        print(f"\nPolishing start #{rank + 1}/{len(top_candidates)} (LHS SSE={lhs_sse:.3f})...", flush=True)
        result = minimize(
            sse_objective, scaled_x0, args=(scaling_factors, observed),
            method="L-BFGS-B", bounds=scaled_bounds,
            options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 100},
        )
        optimized_params = result.x * scaling_factors
        polished_sse = sse_unscaled(optimized_params, observed)
        print(f"  -> polished SSE={polished_sse:.6f}  (iters={result.nit}, evals={result.nfev}, "
              f"x0={optimized_params[IDX_X0]:.3f}, x1={optimized_params[IDX_X1]:.3f})")

        multistart_log.append({
            "rank": rank + 1, "lhs_sse": lhs_sse, "polished_sse": polished_sse,
            "iterations": result.nit, "evaluations": result.nfev,
            "x0": optimized_params[IDX_X0], "x1": optimized_params[IDX_X1],
        })
        if polished_sse < best_polished_sse:
            best_polished_sse, best_result, best_scaling_factors = polished_sse, result, scaling_factors

    pd.DataFrame(multistart_log).to_csv(MULTISTART_LOG_PATH, index=False)
    print(f"\nSaved multi-start log: {MULTISTART_LOG_PATH}")

    best_rank = min(range(len(multistart_log)), key=lambda i: multistart_log[i]["polished_sse"]) + 1
    print(f"\nBest overall: rank #{best_rank} start, polished SSE = {best_polished_sse:.6f}")
    sorted_sse = sorted(r["polished_sse"] for r in multistart_log)
    n_agree = sum(s < sorted_sse[0] * 1.001 for s in sorted_sse)
    print(f"{n_agree} of {len(sorted_sse)} starts reached the best SSE (within 0.1%) -- "
          f"if only 1 did, consider a second run with more samples or another seed.")

    optimized_params = best_result.x * best_scaling_factors
    polished_sse = sse_unscaled(optimized_params, observed)

    # Inverse-Hessian standard errors from L-BFGS-B are a rough approximation and
    # are NOT reliable intervals (they break down near bounds / flat directions).
    # Kept for diagnostics only -- report the 07_uncertainty intervals instead.
    try:
        hess_inv = best_result.hess_inv
        hess_inv = hess_inv if isinstance(hess_inv, np.ndarray) else hess_inv.todense()
        cov_approx = np.asarray(hess_inv) * np.outer(best_scaling_factors, best_scaling_factors)
        std_errors = np.sqrt(np.clip(np.diag(cov_approx), 0, None))
    except (AttributeError, ValueError) as e:
        print(f"\nHessian inverse not usable ({e}).")
        std_errors = np.full(len(PARAMETERS), np.nan)

    df = pd.DataFrame({
        "Parameter": PARAM_NAMES,
        "Bound_lo": [b[0] for b in BOUNDS],
        "Bound_hi": [b[1] for b in BOUNDS],
        "Best_Fit": optimized_params,
        "SE_hessian_approx_unreliable": std_errors,
    })
    print("\n", df.to_string(index=False))
    df.to_csv(os.path.join(OUTPUT_DIR, "vetlyanka_polished_fit.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "vetlyanka_double_summary.txt"), "w") as fh:
        fh.write(f"SSE={polished_sse:.6f}\nk=13\nalignment={'legacy t=0..21' if LEGACY_ALIGNMENT else 't=1..22'}\n"
                 f"samples={os.path.basename(SAMPLES_PATH)}\nTOP_K={TOP_K}\n")
    print(f"\nSaved {OUTPUT_DIR}/vetlyanka_polished_fit.csv")

    gap = optimized_params[IDX_X1] - optimized_params[IDX_X0]
    print(f"x0={optimized_params[IDX_X0]:.3f}, x1={optimized_params[IDX_X1]:.3f}, gap={gap:.3f} weeks")
    if gap < X0_X1_MIN_GAP + 0.05:
        print(f"WARNING: x1 - x0 is on the minimum-gap constraint ({X0_X1_MIN_GAP}) -- "
              f"the constraint is active and the result depends on it.")

    near_bound = [name for name, (lo, hi), val in zip(PARAM_NAMES, BOUNDS, optimized_params)
                  if (val - lo) < 0.05 * (hi - lo) or (hi - val) < 0.05 * (hi - lo)]
    if near_bound:
        print(f"\nWithin 5% of a bound: {near_bound} -- report these as bound-constrained.")
    else:
        print("\nNo fitted parameters are within 5% of a bound.")


if __name__ == "__main__":
    main()