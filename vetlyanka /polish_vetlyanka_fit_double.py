"""
polish_vetlyanka_fit_double.py

Fit for the Vetlyanka double-sigmoid model.

Reads parameter_samples_<NUM_SAMPLES>.csv (from parameter_sampling.py --
regenerate that FIRST if vetlyanka_bounds.py has changed since your last
sample run).

MULTI-START: polishes the TOP_K best LHS candidates independently (not
just the single overall best one) and keeps whichever L-BFGS-B run
converges to the lowest true SSE. A single-start polish can converge
prematurely from a bad starting point without scipy raising any error.

X0/X1 ORDERING CONSTRAINT: candidates where the rise-midpoint (x0) isn't
meaningfully before the decline-onset (x1) are rejected outright. This
was found necessary directly: once b2 and c's bounds were widened, the
optimizer discovered it could build a tall, very gradual rise (high b2,
low c) with x0 pushed out PAST x1, so the decline term does the real
curve-shaping instead of a genuine "rise, peak, THEN a later distinct
decline" -- structurally breaking the two-regime interpretation the
whole model exists to represent, even though it lowers raw SSE. This
was traced to a correlated triple (x0, b2, c) all moving together into
that degenerate configuration -- blocking x0>=x1 removes the escape
route entirely, rather than chasing b2/c bounds that were never the
real constraint.

REQUIRED BEFORE RUNNING:
  1. Set NUM_SAMPLES below to match whichever parameter_samples_*.csv you
     want to search (200000 for the original run this project used, though
     any size works as long as the matching file exists).
  2. If that file doesn't exist yet, or vetlyanka_bounds.py changed since
     it was generated, run parameter_sampling.py first.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from plague_double_sigmoid_model import plague_model
from vetlyanka_bounds import PARAMETERS, PARAM_NAMES, BOUNDS  # single shared source of truth

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "fits", "double")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 200000  # must match an existing data/samples/parameter_samples_<NUM_SAMPLES>.csv
SAMPLES_PATH = os.path.join(SAMPLES_DIR, f"parameter_samples_{NUM_SAMPLES}.csv")
TOP_K = 10  # polish this many independent starting points, keep the best
MULTISTART_LOG_PATH = os.path.join(OUTPUT_DIR, "vetlyanka_double_multistart_log.csv")
X0_X1_MIN_GAP = 1.0  # require x1 - x0 >= this many weeks

t_start, t_end, dt = 0, 22, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
n_weeks = 22
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]  # S0, I0, R0, DI0, DDI0, DR0, P0

# Indices of x0, x1 in PARAM_NAMES / a raw 13-slot parameter row
IDX_X0 = PARAM_NAMES.index("x0")
IDX_X1 = PARAM_NAMES.index("x1")


def scale_parameters(params):
    scaling_factors = np.maximum(np.abs(params), 1e-3)
    return params / scaling_factors, scaling_factors


def load_observed_data():
    """Real Vetlyanka weekly cumulative death data (22 weeks)."""
    cumulativeweekly_all_regions = np.array([
        [3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19, 27, 34,
         90, 259, 313, 345, 364, 376, 376, 376, 376]
    ])
    return cumulativeweekly_all_regions[0].astype(float)


def simulate_cumulative(params):
    sol = solve_ivp(
        plague_model, [t_start, t_end], initial_conditions,
        args=(params,), t_eval=t_points, method="RK45",
    )
    if not sol.success:
        return None
    return sol.y[5][::steps_per_week][:n_weeks]  # DR = cumulative deaths


def sse_unscaled(params, observed):
    """
    Raw SSE, with the x0/x1 ordering constraint enforced first -- see
    module docstring for why this is necessary.
    """
    if params[IDX_X0] >= params[IDX_X1] - X0_X1_MIN_GAP:
        return 1e12

    predicted = simulate_cumulative(params)
    if predicted is None or predicted.shape[0] != observed.shape[0]:
        return 1e12
    return float(np.sum((observed - predicted) ** 2))


def sse_objective(scaled_params, scaling_factors, observed):
    params = scaled_params * scaling_factors
    return sse_unscaled(params, observed)


def find_top_k_lhs_candidates(samples_path: str, observed: np.ndarray, k: int = TOP_K):
    """Load parameter_samples_<NUM_SAMPLES>.csv (columns = PARAM_NAMES, pure
    numeric) and return the k rows with the lowest raw (unscaled) SSE,
    already respecting the x0/x1 ordering constraint via sse_unscaled."""
    samples = np.loadtxt(samples_path, delimiter=",", skiprows=1)
    scored = []
    for i, row in enumerate(samples):
        sse = sse_unscaled(row, observed)
        scored.append((sse, row))
        if (i + 1) % 20000 == 0:
            best_so_far = min(s for s, _ in scored)
            print(f"  scanned {i + 1}/{samples.shape[0]} LHS candidates (best so far: {best_so_far:.3f})")

    scored.sort(key=lambda pair: pair[0])
    top_k = scored[:k]
    print(f"\nTop {k} LHS candidate SSEs: {[f'{s:.3f}' for s, _ in top_k]}")
    return top_k


def main():
    observed = load_observed_data()
    assert observed.shape[0] == n_weeks, (
        f"Expected {n_weeks} weekly observations, got {observed.shape[0]}"
    )

    print(f"Scanning {SAMPLES_PATH} for the top {TOP_K} LHS starting candidates "
          f"(x1 - x0 >= {X0_X1_MIN_GAP} weeks required)...")
    top_candidates = find_top_k_lhs_candidates(SAMPLES_PATH, observed)

    best_result = None
    best_polished_sse = np.inf
    best_scaling_factors = None
    multistart_log = []

    for rank, (lhs_sse, start_row) in enumerate(top_candidates):
        scaled_x0, scaling_factors = scale_parameters(start_row)
        scaled_bounds = [
            (lo / sf if lo is not None else None, hi / sf if hi is not None else None)
            for (lo, hi), sf in zip(BOUNDS, scaling_factors)
        ]

        print(f"\nPolishing start #{rank + 1}/{len(top_candidates)} (LHS SSE={lhs_sse:.3f})...")
        result = minimize(
            sse_objective, scaled_x0,
            args=(scaling_factors, observed),
            method="L-BFGS-B",
            bounds=scaled_bounds,
            options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 100},
        )

        optimized_params = result.x * scaling_factors
        polished_sse = sse_unscaled(optimized_params, observed)
        print(f"  -> polished SSE={polished_sse:.6f}  (iters={result.nit}, evals={result.nfev}, "
              f"x0={optimized_params[IDX_X0]:.3f}, x1={optimized_params[IDX_X1]:.3f})")

        multistart_log.append({
            "rank": rank + 1,
            "lhs_sse": lhs_sse,
            "polished_sse": polished_sse,
            "iterations": result.nit,
            "evaluations": result.nfev,
            "x0": optimized_params[IDX_X0],
            "x1": optimized_params[IDX_X1],
        })

        if polished_sse < best_polished_sse:
            best_polished_sse = polished_sse
            best_result = result
            best_scaling_factors = scaling_factors

    pd.DataFrame(multistart_log).to_csv(MULTISTART_LOG_PATH, index=False)
    print(f"\nSaved multi-start log: {MULTISTART_LOG_PATH}")

    best_rank = min(range(len(multistart_log)), key=lambda i: multistart_log[i]["polished_sse"]) + 1
    print(f"\nBest overall: rank #{best_rank} start, polished SSE = {best_polished_sse:.6f}")
    if best_rank != 1:
        print("NOTE: the single-best-LHS-candidate start was NOT the best after polishing --")
        print("this confirms the space has multiple local optima and single-start polishing")
        print("is not reliable here.")

    result = best_result
    scaling_factors = best_scaling_factors
    optimized_params = result.x * scaling_factors
    polished_sse = sse_unscaled(optimized_params, observed)

    try:
        if isinstance(result.hess_inv, np.ndarray):
            hess_inv_approx = result.hess_inv
        else:
            hess_inv_approx = result.hess_inv.todense()
        hess_inv_approx = np.asarray(hess_inv_approx)
        S_outer = np.outer(scaling_factors, scaling_factors)
        cov_approx = hess_inv_approx * S_outer
        std_errors = np.sqrt(np.diag(cov_approx))
        lower_bounds = np.maximum(optimized_params - 1.96 * std_errors, 0)
        upper_bounds = optimized_params + 1.96 * std_errors
    except (AttributeError, ValueError) as e:
        print(f"\nHessian inverse not usable ({e}). "
              f"Consider bootstrapping for CIs on this fit instead.")
        std_errors = lower_bounds = upper_bounds = np.full(len(PARAMETERS), np.nan)

    df = pd.DataFrame({
        "Parameter": PARAM_NAMES,
        "Bound_lo": [b[0] for b in BOUNDS],
        "Bound_hi": [b[1] for b in BOUNDS],
        "Best_Fit": optimized_params,
        "CI_lower": lower_bounds,
        "CI_upper": upper_bounds,
    })
    print("\n", df.to_string(index=False))
    df.to_csv(os.path.join(OUTPUT_DIR, "vetlyanka_polished_fit.csv"), index=False)
    print(f"\nSaved {OUTPUT_DIR}/vetlyanka_polished_fit.csv")
    print(f"x0={optimized_params[IDX_X0]:.3f}, x1={optimized_params[IDX_X1]:.3f}, "
          f"gap={optimized_params[IDX_X1] - optimized_params[IDX_X0]:.3f} weeks")

    near_bound_flags = []
    for name, (lo, hi), val in zip(PARAM_NAMES, BOUNDS, optimized_params):
        span = hi - lo
        if (val - lo) < 0.05 * span or (hi - val) < 0.05 * span:
            near_bound_flags.append(name)
    if near_bound_flags:
        print(f"\nWARNING: still within 5% of a bound for: {near_bound_flags} "
              f"-- CI on these is likely unreliable; consider whether bounds "
              f"need further widening or whether this reflects a genuinely "
              f"poorly-constrained parameter (worth reporting as such).")
    else:
        print("\nNo fitted parameters are within 5% of a bound.")


if __name__ == "__main__":
    main()