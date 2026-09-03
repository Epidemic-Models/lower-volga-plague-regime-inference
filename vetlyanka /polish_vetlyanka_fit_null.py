"""
polish_vetlyanka_fit_null.py

The "true null" fit: single-sigmoid transmission (rise-only, same as
polish_vetlyanka_fit_single.py) with behavioral feedback DISABLED, not
just re-bounded. sensitivity is fixed at 0 rather than fitted, which
makes exp(-sensitivity*P) == 1 for any P(t) -- the model has no
mechanism left to produce anything but a monotonic rise-and-plateau
curve. T_perceive drops out too, since it only shapes P(t), and P(t) no
longer feeds back into S/I/R once sensitivity=0 (its placeholder value
is provably inert: whatever it's set to, P is multiplied by 0 before it
can affect the ODE, so the simulated trajectory is identical regardless).

9 free parameters: b1, b2, x0, c, gamma1, gamma2, mu1, mu2, dispose_rate.

MULTI-START: polishes the TOP_K best LHS candidates independently and
keeps whichever L-BFGS-B run converges to the lowest true SSE -- added
for consistency with every other fitting script in this project (all of
which turned out to need it), though the stakes here are lower than for
double/single: null already loses by 90-125 AIC points regardless of
small point-estimate shifts, so this is about rigor/consistency, not
about anything close enough to be decided by this fix.

NOTE ON SAMPLING: unlike polish_vetlyanka_fit_double.py, this script draws
its OWN fresh LHS sample below rather than reading a shared
parameter_samples_<N>.csv file. That's intentional, not an oversight: this
model has 9 free parameters, not 13, so the double-sigmoid's sample rows
can't be reused directly the way the single-sigmoid and frozen-gamma/mu
fits reuse them (those keep all 13 slots, just constraining some to be
equal or inert). NUM_LHS_SAMPLES and LHS_SEED below are this script's own,
independent of whatever sample count parameter_sampling.py was last run
with. No x0/x1 ordering constraint is needed either -- this model has no
x1 at all, single-sigmoid-shaped rise only, so it structurally cannot
reach that degenerate configuration.
"""

import numpy as np
import pandas as pd
from pydoe import lhs
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from plague_single_sigmoid_model import plague_model
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "fits", "null")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from vetlyanka_bounds import (
    PARAMETERS_NULL, PARAM_NAMES_NULL, BOUNDS_NULL, K_NULL,
    PARAM_NAMES,  # full 13-name list, for expand_to_full()
)

t_start, t_end, dt = 0, 22, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
n_weeks = 22
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]

NUM_LHS_SAMPLES = 200000
LHS_SEED = 42
TOP_K = 10  # polish this many independent starting points, keep the best
MULTISTART_LOG_PATH = os.path.join(OUTPUT_DIR, "vetlyanka_null_multistart_log.csv")

X1_C1_PLACEHOLDER = 1.0
T_PERCEIVE_PLACEHOLDER = 1.0
SENSITIVITY_FIXED = 0.0  # THE key constraint -- exp(-0*P) == 1, feedback fully disabled


def expand_to_full(null_params):
    null_dict = dict(zip(PARAM_NAMES_NULL, null_params))
    full = []
    for name in PARAM_NAMES:
        if name == 'sensitivity':
            full.append(SENSITIVITY_FIXED)
        elif name == 'T_perceive':
            full.append(T_PERCEIVE_PLACEHOLDER)
        elif name in null_dict:
            full.append(null_dict[name])
        else:  # x1, c1
            full.append(X1_C1_PLACEHOLDER)
    return np.array(full)


def load_observed_data():
    cumulativeweekly_all_regions = np.array([
        [3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19, 27, 34,
         90, 259, 313, 345, 364, 376, 376, 376, 376]
    ])
    return cumulativeweekly_all_regions[0].astype(float)


def simulate_cumulative(null_params):
    full_params = expand_to_full(null_params)
    sol = solve_ivp(
        plague_model, [t_start, t_end], initial_conditions,
        args=(full_params,), t_eval=t_points, method="RK45",
    )
    if not sol.success:
        return None
    return sol.y[5][::steps_per_week][:n_weeks]


def sse_unscaled(null_params, observed):
    predicted = simulate_cumulative(null_params)
    if predicted is None or predicted.shape[0] != observed.shape[0]:
        return 1e12
    sse = np.sum((observed - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def sse_objective(scaled_params, scaling_factors, observed):
    return sse_unscaled(scaled_params * scaling_factors, observed)


def scale_parameters(params):
    scaling_factors = np.maximum(np.abs(params), 1e-3)
    return params / scaling_factors, scaling_factors


def calculate_aic(sse, n, k):
    return float(n * np.log(sse / n) + 2 * k) if sse > 0 and np.isfinite(sse) else np.nan


def calculate_aicc(sse, n, k):
    aic = calculate_aic(sse, n, k)
    denom = n - k - 1
    return float(aic + (2 * k * (k + 1)) / denom) if np.isfinite(aic) and denom > 0 else np.nan


def find_top_k_lhs_candidates(observed, k=TOP_K):
    num_params = len(PARAMETERS_NULL)
    print(f"Running LHS search ({NUM_LHS_SAMPLES} samples, seed={LHS_SEED}) "
          f"over {num_params} true-null parameters (feedback disabled), "
          f"scoring all candidates to find the top {k}...")
    np.random.seed(LHS_SEED)
    lhs_unit = lhs(num_params, samples=NUM_LHS_SAMPLES)
    lhs_samples = np.array([
        [row[j] * (BOUNDS_NULL[j][1] - BOUNDS_NULL[j][0]) + BOUNDS_NULL[j][0]
         for j in range(num_params)]
        for row in lhs_unit
    ])

    scored = []
    for i, row in enumerate(lhs_samples):
        sse = sse_unscaled(row, observed)
        scored.append((sse, row.copy()))
        if (i + 1) % 20000 == 0:
            best_so_far = min(s for s, _ in scored)
            print(f"  LHS sample {i + 1}/{NUM_LHS_SAMPLES}  (best so far: {best_so_far:.3f})")

    scored.sort(key=lambda pair: pair[0])
    top_k = scored[:k]
    print(f"\nTop {k} LHS candidate SSEs: {[f'{s:.3f}' for s, _ in top_k]}")
    return top_k


def main():
    observed = load_observed_data()

    top_candidates = find_top_k_lhs_candidates(observed)

    best_result = None
    best_polished_sse = np.inf
    best_scaling_factors = None
    multistart_log = []

    for rank, (lhs_sse, start_row) in enumerate(top_candidates):
        scaled_x0, scaling_factors = scale_parameters(start_row)
        scaled_bounds = [(lo / sf, hi / sf) for (lo, hi), sf in zip(BOUNDS_NULL, scaling_factors)]

        print(f"\nPolishing start #{rank + 1}/{TOP_K} (LHS SSE={lhs_sse:.3f})...")
        result = minimize(sse_objective, scaled_x0, args=(scaling_factors, observed),
                           method="L-BFGS-B", bounds=scaled_bounds,
                           options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 100})

        optimized = result.x * scaling_factors
        polished_sse = sse_unscaled(optimized, observed)
        print(f"  -> polished SSE={polished_sse:.6f}  (iters={result.nit}, evals={result.nfev})")

        multistart_log.append({
            "rank": rank + 1, "lhs_sse": lhs_sse, "polished_sse": polished_sse,
            "iterations": result.nit, "evaluations": result.nfev,
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
        print("NOTE: the single-best-LHS-candidate start was NOT the best after polishing.")

    optimized_params = best_result.x * best_scaling_factors
    polished_sse = sse_unscaled(optimized_params, observed)
    aic = calculate_aic(polished_sse, n_weeks, K_NULL)
    aicc = calculate_aicc(polished_sse, n_weeks, K_NULL)

    try:
        if isinstance(best_result.hess_inv, np.ndarray):
            hess_inv_approx = best_result.hess_inv
        else:
            hess_inv_approx = best_result.hess_inv.todense()
        hess_inv_approx = np.asarray(hess_inv_approx)
        S_outer = np.outer(best_scaling_factors, best_scaling_factors)
        cov_approx = hess_inv_approx * S_outer
        std_errors = np.sqrt(np.diag(cov_approx))
        lower_bounds = np.maximum(optimized_params - 1.96 * std_errors, 0)
        upper_bounds = optimized_params + 1.96 * std_errors
    except (AttributeError, ValueError) as e:
        print(f"\nHessian inverse not usable ({e}). Consider bootstrapping for CIs instead.")
        std_errors = lower_bounds = upper_bounds = np.full(K_NULL, np.nan)

    df = pd.DataFrame({
        "Parameter": PARAM_NAMES_NULL,
        "Bound_lo": [b[0] for b in BOUNDS_NULL],
        "Bound_hi": [b[1] for b in BOUNDS_NULL],
        "Best_Fit": optimized_params,
        "CI_lower": lower_bounds,
        "CI_upper": upper_bounds,
    })
    print("\n", df.to_string(index=False))
    df.to_csv(os.path.join(OUTPUT_DIR, "vetlyanka_polished_fit_null.csv"), index=False)
    print(f"\nSaved {OUTPUT_DIR}/vetlyanka_polished_fit_null.csv")

    print(f"\nPolished SSE: {polished_sse:.6f}")
    print(f"AIC: {aic:.6f}   AICc: {aicc:.6f}")

    near_bound_flags = []
    for name, (lo, hi), val in zip(PARAM_NAMES_NULL, BOUNDS_NULL, optimized_params):
        span = hi - lo
        if (val - lo) < 0.05 * span or (hi - val) < 0.05 * span:
            near_bound_flags.append(name)
    if near_bound_flags:
        print(f"\nWARNING: still within 5% of a bound for: {near_bound_flags} "
              f"-- CI on these is likely unreliable.")
    else:
        print("\nNo fitted parameters are within 5% of a bound.")


if __name__ == "__main__":
    main()