"""
single_frozen_T_perceive_sensitivity_check.py

Sensitivity check for T_perceive in the single-frozen model, extended
this session to include short-lag values (0.1, 0.3, 0.5) alongside the
originally-tested mid-to-long range. Motivated directly by Malta's
finding: single-sigmoid genuinely outperforms double-sigmoid there at
short perception-lag values (T_perceive<0.5), a real, solver-confirmed
result, not an artifact. Vetlyanka's own single-frozen T_perceive
always pinned at the UPPER wall when searched freely (wanting to go
LONGER, never shorter), which is informative but not the same as a
direct check of the short-lag region -- this script closes that gap.

For the short-lag values specifically, this version ALSO runs a
solver-robustness check (default RK45, RK45 tightened 10,000x, Radau)
on the resulting best-fit parameters, matching exactly the check that
confirmed Malta's crossover was genuine -- so if Vetlyanka shows
anything similar at short lag, it can be trusted or ruled out with the
same rigor, not left as an open question.

WHY THIS EXISTS: single-frozen's real fit fixes T_perceive at a
constant (polish_vetlyanka_fit_single_freeze_mu_gamma_FIXED.py uses
3.409, correctly family-matched to double-frozen's own converged
value) rather than searching it, because it pinned at every ceiling
tried with only trivial SSE gain -- the signature of a non-converging
parameter. This script tests whether that conclusion (single-frozen
loses to double-frozen) holds across the FULL plausible range,
including the short-lag region motivated by Malta's finding.

DATA LAYOUT: reads/writes from this project's data/ folder.

Outputs:
    single_frozen_T_perceive_sensitivity_results.csv  -- one row per
        T_perceive value tested, with its best SSE and full parameter fit
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
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

t_start, t_end, dt = 0.0, 22.0, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
n_weeks = 22
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]

NUM_LHS_SAMPLES = 100000
LHS_SEED = 42
TOP_K = 5
X1_C1_PLACEHOLDER = 1.0

DOUBLE_FROZEN_SSE = 411.978  # current locked double-frozen result -- must match
                              # the real, final value, not an earlier draft SSE

# Extended range: short-lag values (0.1, 0.3, 0.5) added this session,
# motivated by Malta's confirmed short-lag crossover, alongside the
# originally-tested mid-to-long range.
T_PERCEIVE_TEST_VALUES = [0.1, 0.3, 0.5, 2.5, 3.486, 5.0, 6.59, 8.0]
SHORT_LAG_VALUES = [0.1, 0.3, 0.5]  # these get the extra solver-robustness check

# ---- Same indices/bounds construction as the real single-frozen script ----
IDX_GAMMA1, IDX_GAMMA2, IDX_MU1, IDX_MU2 = 4, 5, 6, 7

GAMMA_CONST_BOUND = (
    min(BOUNDS_SINGLE[IDX_GAMMA1][0], BOUNDS_SINGLE[IDX_GAMMA2][0]),
    max(BOUNDS_SINGLE[IDX_GAMMA1][1], BOUNDS_SINGLE[IDX_GAMMA2][1]),
)
MU_CONST_BOUND = (
    min(BOUNDS_SINGLE[IDX_MU1][0], BOUNDS_SINGLE[IDX_MU2][0]),
    max(BOUNDS_SINGLE[IDX_MU1][1], BOUNDS_SINGLE[IDX_MU2][1]),
)

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
    elif PARAM_NAMES_SINGLE[i] == "T_perceive":
        continue  # fixed per test value, not searched
    else:
        REDUCED_PARAM_NAMES.append(name)
        REDUCED_BOUNDS.append(bound)

K_REDUCED = len(REDUCED_PARAM_NAMES)
IDX_T_PERCEIVE_SINGLE = PARAM_NAMES_SINGLE.index("T_perceive")


def load_observed_data():
    return np.array([3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19,
                      27, 34, 90, 259, 313, 345, 364, 376,
                      376, 376, 376], dtype=float)


def reduced_to_full(reduced_params, t_perceive_fixed):
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
        elif i == IDX_T_PERCEIVE_SINGLE:
            single_vector[IDX_T_PERCEIVE_SINGLE] = t_perceive_fixed
            continue
        else:
            single_vector[i] = reduced_params[r_idx]
            r_idx += 1
    single_dict = dict(zip(PARAM_NAMES_SINGLE, single_vector))
    return np.array([single_dict.get(name, X1_C1_PLACEHOLDER) for name in PARAM_NAMES], dtype=float)


def simulate_cumulative(reduced_params, t_perceive_fixed, solver_kwargs=None):
    full_params = reduced_to_full(reduced_params, t_perceive_fixed)
    kwargs = solver_kwargs or {"method": "RK45"}
    try:
        sol = solve_ivp(plague_model, [t_start, t_end], initial_conditions,
                         args=(full_params,), t_eval=t_points, **kwargs)
    except (ValueError, FloatingPointError, OverflowError):
        return None
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    predicted = sol.y[5][::steps_per_week][:n_weeks]
    if predicted.shape[0] != n_weeks or not np.all(np.isfinite(predicted)):
        return None
    return predicted


def sse_unscaled(reduced_params, observed, t_perceive_fixed):
    predicted = simulate_cumulative(reduced_params, t_perceive_fixed)
    if predicted is None:
        return 1e12
    sse = np.sum((observed - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def sse_objective(scaled_params, scaling_factors, observed, t_perceive_fixed):
    return sse_unscaled(scaled_params * scaling_factors, observed, t_perceive_fixed)


def scale_parameters(params):
    scaling_factors = np.maximum(np.abs(params), 1e-3)
    return params / scaling_factors, scaling_factors


def fit_at_fixed_T_perceive(t_perceive_fixed, observed):
    print(f"\n{'='*60}\nFitting single-frozen with T_perceive FIXED at {t_perceive_fixed}\n{'='*60}")

    np.random.seed(LHS_SEED)
    lhs_unit = lhs(K_REDUCED, samples=NUM_LHS_SAMPLES)
    lhs_samples = np.array([
        [row[j] * (REDUCED_BOUNDS[j][1] - REDUCED_BOUNDS[j][0]) + REDUCED_BOUNDS[j][0]
         for j in range(K_REDUCED)]
        for row in lhs_unit
    ])

    scored = []
    for i, row in enumerate(lhs_samples):
        sse = sse_unscaled(row, observed, t_perceive_fixed)
        scored.append((sse, row.copy()))
        if (i + 1) % 25000 == 0:
            best_so_far = min(s for s, _ in scored)
            print(f"  scanned {i + 1}/{NUM_LHS_SAMPLES}  (best so far: {best_so_far:.3f})")

    scored.sort(key=lambda pair: pair[0])
    top_candidates = scored[:TOP_K]

    best_polished_sse = np.inf
    best_optimized = None

    for rank, (lhs_sse, start_row) in enumerate(top_candidates):
        scaled_x0, scaling_factors = scale_parameters(start_row)
        scaled_bounds = [(lo / sf, hi / sf) for (lo, hi), sf in zip(REDUCED_BOUNDS, scaling_factors)]

        result = minimize(sse_objective, scaled_x0, args=(scaling_factors, observed, t_perceive_fixed),
                           method="L-BFGS-B", bounds=scaled_bounds,
                           options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 100})

        optimized = result.x * scaling_factors
        polished_sse = sse_unscaled(optimized, observed, t_perceive_fixed)
        print(f"  start #{rank + 1}/{TOP_K}: LHS SSE={lhs_sse:.3f} -> polished SSE={polished_sse:.6f}")

        if polished_sse < best_polished_sse:
            best_polished_sse = polished_sse
            best_optimized = optimized

    print(f"Best SSE at T_perceive={t_perceive_fixed}: {best_polished_sse:.6f}")
    return best_polished_sse, best_optimized


def solver_robustness_check(best_reduced, t_perceive_fixed, observed):
    """Re-solve the SAME best-fit parameters under three solver
    configurations -- matches malta_crossover_not_numerical_artefact.py."""
    runs = [
        ("Default RK45 (loose tolerances)", {"method": "RK45"}),
        ("RK45, tolerances tightened 10,000x", {"method": "RK45", "rtol": 1e-10, "atol": 1e-12}),
        ("Radau (implicit stiff-solver), same tight tolerances", {"method": "Radau", "rtol": 1e-10, "atol": 1e-12}),
    ]
    print(f"\n  Solver-robustness check at T_perceive={t_perceive_fixed}:")
    sses = []
    for label, kwargs in runs:
        predicted = simulate_cumulative(best_reduced, t_perceive_fixed, solver_kwargs=kwargs)
        if predicted is None:
            print(f"    {label:<50} SOLVE FAILED")
            sses.append(None)
            continue
        sse = float(np.sum((observed - predicted) ** 2))
        print(f"    {label:<50} SSE = {sse:.3f}")
        sses.append(sse)
    valid = [s for s in sses if s is not None]
    if len(valid) >= 2:
        spread = max(valid) - min(valid)
        status = "STABLE" if spread < 0.01 * min(valid) else "UNSTABLE -- WARNING"
        print(f"  SSE spread across solvers: {spread:.3f} ({status})")
    return sses


def main():
    observed = load_observed_data()

    results = []
    for t_perceive_fixed in T_PERCEIVE_TEST_VALUES:
        best_sse, best_params = fit_at_fixed_T_perceive(t_perceive_fixed, observed)
        row = {"T_perceive_fixed": t_perceive_fixed, "Best_SSE": best_sse,
               "vs_double_frozen": "single still loses" if best_sse > DOUBLE_FROZEN_SSE else "single WINS -- investigate"}
        for name, val in zip(REDUCED_PARAM_NAMES, best_params):
            row[name] = val
        results.append(row)

        if t_perceive_fixed in SHORT_LAG_VALUES:
            solver_robustness_check(best_params, t_perceive_fixed, observed)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "single_frozen_T_perceive_sensitivity_results.csv"), index=False)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Double-frozen SSE (fixed reference): {DOUBLE_FROZEN_SSE}")
    print(results_df[["T_perceive_fixed", "Best_SSE", "vs_double_frozen"]].to_string(index=False))

    if (results_df["Best_SSE"] > DOUBLE_FROZEN_SSE).all():
        print("\nCONFIRMED: single-frozen loses to double-frozen at EVERY T_perceive value")
        print("tested, INCLUDING the short-lag region motivated by Malta's finding.")
        print("The model comparison conclusion does not depend on the specific fixed")
        print("constant chosen, across the full plausible range.")
    else:
        print("\nWARNING: single-frozen beat double-frozen at one or more T_perceive")
        print("values -- this needs investigation, matching what was found for Malta,")
        print("before reporting the constant-T_perceive result as robust.")

    print(f"\nSaved {OUTPUT_DIR}/single_frozen_T_perceive_sensitivity_results.csv")


if __name__ == "__main__":
    main()