"""
polish_vetlyanka_fit_single_freeze_mu_gamma_FIXED.py

Corrected version: real PARAM_NAMES_SINGLE indices are
  0 b1, 1 b2, 2 x0, 3 c, 4 gamma1, 5 gamma2,
  6 mu1, 7 mu2, 8 T_perceive, 9 sensitivity, 10 dispose_rate
(no c1 slot -- that only exists in the 13-parameter double-sigmoid list).

MULTI-START: polishes the TOP_K best LHS candidates independently (not
just the single overall best one) and keeps whichever L-BFGS-B run
converges to the lowest true SSE.

T_PERCEIVE FIXED, NOT SEARCHED: across three rounds (bound ceilings 8,
10, and 25), T_perceive never once converged to an interior value --
it pinned at whatever ceiling it was given, with SSE improving only
trivially even as the ceiling was raised substantially (663 at ceiling
25 vs 709 at ceiling 10 -- tightening the bound made the fit WORSE while
it still pinned dead at the new wall). That combination (worse fit, still
pinned) is the signature of a parameter that isn't converging, not one
that's under-bounded -- unlike x0 in the double-sigmoid model, which
genuinely improved substantially once given real room. Rather than widen
indefinitely, T_perceive is fixed here at 6.59 weeks, the already-
validated interior value from the original sigmoidal single-sigmoid fit
(polish_vetlyanka_fit_single.py), where it converged cleanly under a much
more modest (0.5, 8) bound. This is the same treatment as sensitivity=0
in the null model: a parameter fixed by argument/evidence rather than
searched.

NOTE: this model has no x1 (decline-onset) parameter at all -- its
beta(t) is rise-only -- so it cannot reach the x0>=x1 degenerate
configuration the double-sigmoid scripts guard against. No equivalent
constraint is needed here.
"""

import os
import numpy as np
import pandas as pd

from pydoe import lhs
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from plague_single_sigmoid_model import plague_model
from vetlyanka_bounds import PARAM_NAMES_SINGLE, BOUNDS_SINGLE, PARAM_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "fits", "single_frozen")
os.makedirs(OUTPUT_DIR, exist_ok=True)

t_start, t_end, dt = 0.0, 22.0, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
n_weeks = 22
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]

NUM_LHS_SAMPLES = 200000
LHS_SEED = 42
TOP_K = 10  # polish this many independent starting points, keep the best
X1_C1_PLACEHOLDER = 1.0

# ---- FIXED indices, matching the real PARAM_NAMES_SINGLE ordering ----
IDX_GAMMA1 = 4
IDX_GAMMA2 = 5
IDX_MU1 = 6
IDX_MU2 = 7
IDX_T_PERCEIVE = 8
T_PERCEIVE_FIXED = 3.409  # frozen-double's own converged value, correct family match

# Span (union), not average -- same fix as the double-sigmoid scripts.
GAMMA_CONST_BOUND = (
    min(BOUNDS_SINGLE[IDX_GAMMA1][0], BOUNDS_SINGLE[IDX_GAMMA2][0]),
    max(BOUNDS_SINGLE[IDX_GAMMA1][1], BOUNDS_SINGLE[IDX_GAMMA2][1]),
)
MU_CONST_BOUND = (
    min(BOUNDS_SINGLE[IDX_MU1][0], BOUNDS_SINGLE[IDX_MU2][0]),
    max(BOUNDS_SINGLE[IDX_MU1][1], BOUNDS_SINGLE[IDX_MU2][1]),
)
print(f"GAMMA_CONST_BOUND = {GAMMA_CONST_BOUND}")
print(f"MU_CONST_BOUND    = {MU_CONST_BOUND}")
print(f"T_perceive fixed at {T_PERCEIVE_FIXED} weeks (not searched)")

REDUCED_PARAM_NAMES = []
REDUCED_BOUNDS = []
for i, (name, bound) in enumerate(zip(PARAM_NAMES_SINGLE, BOUNDS_SINGLE)):
    if i == IDX_GAMMA1:
        REDUCED_PARAM_NAMES.append("gamma_const")
        REDUCED_BOUNDS.append(GAMMA_CONST_BOUND)
    elif i == IDX_GAMMA2:
        continue
    elif i == IDX_MU1:
        REDUCED_PARAM_NAMES.append("mu_const")
        REDUCED_BOUNDS.append(MU_CONST_BOUND)
    elif i == IDX_MU2:
        continue
    elif i == IDX_T_PERCEIVE:
        continue  # fixed, not searched
    else:
        REDUCED_PARAM_NAMES.append(name)
        REDUCED_BOUNDS.append(bound)

K_REDUCED = len(REDUCED_PARAM_NAMES)
print(f"\nReduced parameters ({K_REDUCED}):")
for name, bound in zip(REDUCED_PARAM_NAMES, REDUCED_BOUNDS):
    print(f"  {name:15s} {bound}")


def load_observed_data():
    return np.array([3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19,
                      27, 34, 90, 259, 313, 345, 364, 376,
                      376, 376, 376], dtype=float)


def reduced_to_full(reduced_params):
    single_vector = np.zeros(len(PARAM_NAMES_SINGLE), dtype=float)
    r_idx = 0
    for i, name in enumerate(PARAM_NAMES_SINGLE):
        if i == IDX_GAMMA1:
            gamma_const = reduced_params[r_idx]
            single_vector[IDX_GAMMA1] = gamma_const
            single_vector[IDX_GAMMA2] = gamma_const
            r_idx += 1
        elif i == IDX_GAMMA2:
            continue
        elif i == IDX_MU1:
            mu_const = reduced_params[r_idx]
            single_vector[IDX_MU1] = mu_const
            single_vector[IDX_MU2] = mu_const
            r_idx += 1
        elif i == IDX_MU2:
            continue
        elif i == IDX_T_PERCEIVE:
            single_vector[IDX_T_PERCEIVE] = T_PERCEIVE_FIXED  # fixed value,
            continue                                          # consumes no slot
        else:
            single_vector[i] = reduced_params[r_idx]
            r_idx += 1
    single_dict = dict(zip(PARAM_NAMES_SINGLE, single_vector))
    return np.array([single_dict.get(name, X1_C1_PLACEHOLDER) for name in PARAM_NAMES], dtype=float)


def simulate_cumulative(reduced_params):
    full_params = reduced_to_full(reduced_params)
    try:
        sol = solve_ivp(plague_model, [t_start, t_end], initial_conditions,
                         args=(full_params,), t_eval=t_points, method="RK45")
    except (ValueError, FloatingPointError, OverflowError):
        return None
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    predicted = sol.y[5][::steps_per_week][:n_weeks]
    if predicted.shape[0] != n_weeks or not np.all(np.isfinite(predicted)):
        return None
    return predicted


def sse_unscaled(reduced_params, observed):
    predicted = simulate_cumulative(reduced_params)
    if predicted is None:
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


def main():
    observed = load_observed_data()
    np.random.seed(LHS_SEED)
    lhs_unit = lhs(K_REDUCED, samples=NUM_LHS_SAMPLES)
    lhs_samples = np.array([
        [row[j] * (REDUCED_BOUNDS[j][1] - REDUCED_BOUNDS[j][0]) + REDUCED_BOUNDS[j][0]
         for j in range(K_REDUCED)]
        for row in lhs_unit
    ])

    print(f"\nRunning LHS search ({NUM_LHS_SAMPLES} samples), scoring all candidates "
          f"to find the top {TOP_K}...")
    scored = []
    for i, row in enumerate(lhs_samples):
        if i > 0 and i % 40000 == 0:
            best_so_far = min(s for s, _ in scored)
            print(f"  {i}/{NUM_LHS_SAMPLES}  best so far: {best_so_far:.3f}")
        sse = sse_unscaled(row, observed)
        scored.append((sse, row.copy()))

    scored.sort(key=lambda pair: pair[0])
    top_candidates = scored[:TOP_K]
    print(f"\nTop {TOP_K} LHS candidate SSEs: {[f'{s:.3f}' for s, _ in top_candidates]}")

    best_result = None
    best_polished_sse = np.inf
    best_scaling_factors = None
    multistart_log = []

    for rank, (lhs_sse, start_row) in enumerate(top_candidates):
        scaled_x0, scaling_factors = scale_parameters(start_row)
        scaled_bounds = [(lo / sf, hi / sf) for (lo, hi), sf in zip(REDUCED_BOUNDS, scaling_factors)]

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

    log_path = os.path.join(OUTPUT_DIR, "vetlyanka_constant_mu_gamma_single_multistart_log.csv")
    pd.DataFrame(multistart_log).to_csv(log_path, index=False)
    print(f"\nSaved multi-start log: {log_path}")

    best_rank = min(range(len(multistart_log)), key=lambda i: multistart_log[i]["polished_sse"]) + 1
    print(f"\nBest overall: rank #{best_rank} start, polished SSE = {best_polished_sse:.6f}")
    if best_rank != 1:
        print("NOTE: the single-best-LHS-candidate start was NOT the best after polishing.")

    optimized = best_result.x * best_scaling_factors
    polished_sse = sse_unscaled(optimized, observed)
    aic = calculate_aic(polished_sse, n_weeks, K_REDUCED)
    aicc = calculate_aicc(polished_sse, n_weeks, K_REDUCED)

    print(f"\nFinal polished SSE: {polished_sse:.6f}")
    print(f"AIC: {aic:.6f}   AICc: {aicc:.6f}")
    print(f"(T_perceive fixed at {T_PERCEIVE_FIXED}, not counted as a free parameter -- "
          f"K_REDUCED={K_REDUCED})")
    print("\nBest-fit reduced parameters:")
    for name, val in zip(REDUCED_PARAM_NAMES, optimized):
        lo, hi = dict(zip(REDUCED_PARAM_NAMES, REDUCED_BOUNDS))[name]
        pct = (val - lo) / (hi - lo) * 100
        flag = "  <-- near wall" if (pct < 5 or pct > 95) else ""
        print(f"  {name:15s} = {val:.6f}  ({pct:.1f}% into bound){flag}")

    fit_df = pd.DataFrame({
        "Parameter": REDUCED_PARAM_NAMES,
        "Bound_lo": [b[0] for b in REDUCED_BOUNDS],
        "Bound_hi": [b[1] for b in REDUCED_BOUNDS],
        "Best_Fit": optimized,
    })
    fit_df.to_csv(os.path.join(OUTPUT_DIR, "vetlyanka_constant_mu_gamma_single_FIXED_fit.csv"), index=False)
    print(f"\nSaved {OUTPUT_DIR}/vetlyanka_constant_mu_gamma_single_FIXED_fit.csv")
    print(f"\nNOTE: T_perceive is fixed at {T_PERCEIVE_FIXED} and will NOT appear as a row in "
          f"this CSV -- downstream scripts (AIC_AICc_BIC.py, frozen_confidence_intervals_single.py, "
          f"etc.) that read T_perceive from this file need updating to use the fixed constant instead.")


if __name__ == "__main__":
    main()