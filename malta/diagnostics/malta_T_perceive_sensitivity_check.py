"""
malta_T_perceive_sensitivity_check.py

Standalone sensitivity check for Malta's T_perceive, testing BOTH
single-sigmoid and double-sigmoid (unlike Vetlyanka, where only
single-sigmoid needed this -- Malta's double-sigmoid also pinned
T_perceive at every widened bound tried, so there's no clean,
non-pinning model to anchor a fixed constant from here).

Does NOT modify malta_fit_single.py or malta_fit_double.py -- this is a
separate script with its own independent LHS search, T_perceive fixed
at each test value in turn rather than searched.

For each T_perceive value tested, both models are refit (multi-start:
LHS scan + L-BFGS-B polish) with T_perceive held constant, over the
remaining parameters only.

Required:
    plague_single_sigmoid_model.py and plague_double_sigmoid_model.py in
    the vetlyanka/ folder (imported via an explicit path below).
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import qmc

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # malta/, now that this sits in diagnostics/
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "malta_T_perceive_sensitivity.csv")

VETLYANKA_DIR = "/Users/bornalid/PycharmProjects/lower-volga-plague-regime-inference/vetlyanka "
sys.path.insert(0, VETLYANKA_DIR)

from plague_single_sigmoid_model import plague_model as plague_model_single
from plague_double_sigmoid_model import plague_model as plague_model_double

# ------------------------------------------------------------
# Malta data and widened bounds (matching malta_fit_single.py /
# malta_fit_double.py's already-widened versions)
# ------------------------------------------------------------
observed = np.array([0, 0, 0, 41, 100, 227, 440, 592, 813, 1139, 1523, 1896,
                      2225, 2609, 2860, 3045, 3269, 3486, 3703, 3902, 4045,
                      4147, 4247, 4309, 4339, 4379, 4400], dtype=float)
n_weeks = len(observed)
t_start, t_end, dt = 0, n_weeks - 1, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
initial_conditions = [11000, 1, 0, 0, 0, 0, 0]

T_PERCEIVE_TEST_VALUES = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
NUM_LHS_SAMPLES = 20000
TOP_K = 5
X0_X1_MIN_GAP = 1.0

# Single-sigmoid: 10 params excluding T_perceive
PARAM_NAMES_SINGLE = ["b1", "b2", "x0", "c", "gamma1", "gamma2", "mu1", "mu2", "sensitivity", "dispose_rate"]
BOUNDS_SINGLE = [(0.01, 0.2), (5.00, 20.00), (3.5, 15), (0.02, 2.0), (0.35, 0.7),
                 (1.0, 1.4), (1.0, 1.4), (1.0, 10), (0.005, 0.05), (2.0, 7.5)]
FULL_NAMES_SINGLE = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2",
                     "mu1", "mu2", "T_perceive", "sensitivity", "dispose_rate"]

# Double-sigmoid: 12 params excluding T_perceive
PARAM_NAMES_DOUBLE = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2",
                      "mu1", "mu2", "sensitivity", "dispose_rate"]
BOUNDS_DOUBLE = [(0.01, 0.2), (5.00, 20.00), (3.5, 15), (17, 21), (0.02, 2.0), (0.8, 4.0),
                 (0.35, 0.7), (1.0, 1.4), (1.0, 1.4), (1.0, 10), (0.005, 0.05), (2.0, 7.5)]
IDX_X0_D = PARAM_NAMES_DOUBLE.index("x0")
IDX_X1_D = PARAM_NAMES_DOUBLE.index("x1")


def simulate_weekly(model_fn, params):
    try:
        sol = solve_ivp(model_fn, [t_start, t_end], initial_conditions,
                         args=(params,), t_eval=t_points, method="RK45")
    except (ValueError, FloatingPointError, OverflowError):
        return None
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    predicted = sol.y[5][::steps_per_week][:n_weeks]
    if predicted.shape[0] != n_weeks or not np.all(np.isfinite(predicted)):
        return None
    return predicted


def sse_single(reduced, t_perceive_fixed):
    d = dict(zip(PARAM_NAMES_SINGLE, reduced))
    d["T_perceive"] = t_perceive_fixed
    full = np.array([d.get(n, 1.0) for n in FULL_NAMES_SINGLE])
    predicted = simulate_weekly(plague_model_single, full)
    if predicted is None:
        return 1e12
    sse = np.sum((observed - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def sse_double(reduced, t_perceive_fixed):
    if reduced[IDX_X0_D] >= reduced[IDX_X1_D] - X0_X1_MIN_GAP:
        return 1e12
    d = dict(zip(PARAM_NAMES_DOUBLE, reduced))
    full = np.array([d["b1"], d["b2"], d["x0"], d["x1"], d["c"], d["c1"],
                      d["gamma1"], d["gamma2"], d["mu1"], d["mu2"],
                      t_perceive_fixed, d["sensitivity"], d["dispose_rate"]])
    predicted = simulate_weekly(plague_model_double, full)
    if predicted is None:
        return 1e12
    sse = np.sum((observed - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def scale_parameters(params):
    scaling_factors = np.maximum(np.abs(params), 1e-3)
    return params / scaling_factors, scaling_factors


def fit_at_fixed_t_perceive(sse_fn, param_names, bounds, t_perceive_fixed, seed):
    K = len(param_names)
    sampler = qmc.LatinHypercube(d=K, seed=seed)
    unit = sampler.random(n=NUM_LHS_SAMPLES)
    lhs_samples = np.array([
        [row[j] * (bounds[j][1] - bounds[j][0]) + bounds[j][0] for j in range(K)]
        for row in unit
    ])
    scored = [(sse_fn(row, t_perceive_fixed), row) for row in lhs_samples]
    scored.sort(key=lambda pair: pair[0])
    top_k = scored[:TOP_K]

    best_sse = np.inf
    for lhs_sse, start_row in top_k:
        scaled_x0, scaling_factors = scale_parameters(start_row)
        scaled_bounds = [(lo / sf, hi / sf) for (lo, hi), sf in zip(bounds, scaling_factors)]
        result = minimize(lambda sp: sse_fn(sp * scaling_factors, t_perceive_fixed),
                           scaled_x0, method="L-BFGS-B", bounds=scaled_bounds,
                           options={"maxiter": 5000, "ftol": 1e-10, "gtol": 1e-8})
        optimized = result.x * scaling_factors
        polished_sse = sse_fn(optimized, t_perceive_fixed)
        if polished_sse < best_sse:
            best_sse = polished_sse
    return best_sse


def main():
    print(f"{'T_perceive':>12} {'Single SSE':>14} {'Double SSE':>14}")
    print("-" * 42)
    results = []
    for t_val in T_PERCEIVE_TEST_VALUES:
        s_sse = fit_at_fixed_t_perceive(sse_single, PARAM_NAMES_SINGLE, BOUNDS_SINGLE, t_val, seed=42)
        d_sse = fit_at_fixed_t_perceive(sse_double, PARAM_NAMES_DOUBLE, BOUNDS_DOUBLE, t_val, seed=42)
        print(f"{t_val:>12.2f} {s_sse:>14.2f} {d_sse:>14.2f}")
        results.append({"T_perceive": t_val, "single_SSE": s_sse, "double_SSE": d_sse})

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {OUTPUT_CSV}")

    print("\nInterpretation:")
    print("- If SSE keeps improving toward the low end with no sign of leveling off,")
    print("  that's the same 'genuinely non-converging' signature found in Vetlyanka --")
    print("  don't chase it further, fix T_perceive at whichever value is defensible")
    print("  and report the conclusion's robustness across this whole table instead.")
    print("- If double consistently beats single across every value tested, the model")
    print("  comparison conclusion doesn't depend on which T_perceive was chosen,")
    print("  even without a clean, non-pinning anchor value to fix it at.")


if __name__ == "__main__":
    main()