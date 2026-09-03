"""
malta_crossover_not_numerical_artefact.py

Confirms whether Malta's T_perceive sensitivity crossover (single beats
double at T_perceive=0.1 and 0.3, unlike every other tested value where
double wins) is a genuine result or a numerical-integration artifact.
Console RuntimeWarnings (overflow, invalid value) clustered specifically
in this low-T_perceive region during the original sensitivity sweep,
which is the direct reason to check this before trusting the crossover.

Same logic as check_frozen_rebound_not_numerical_artefact.py: refit both
models at each crossover T_perceive value, then re-solve the resulting
best-fit parameters under three solver configurations (default RK45,
RK45 with tolerances tightened 10,000x, and Radau, an implicit method
suited to stiff systems). If SSE agrees closely across all three, the
crossover is genuine, not an artifact.

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

VETLYANKA_DIR = "/Users/bornalid/PycharmProjects/lower-volga-plague-regime-inference/vetlyanka "
sys.path.insert(0, VETLYANKA_DIR)

from plague_single_sigmoid_model import plague_model as plague_model_single
from plague_double_sigmoid_model import plague_model as plague_model_double

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # malta/, once this sits in diagnostics/
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "malta_crossover_not_numerical_artefact.csv")

# ------------------------------------------------------------
# Malta data and widened bounds (matching malta_fit_single.py /
# malta_fit_double.py / malta_T_perceive_sensitivity_check.py)
# ------------------------------------------------------------
observed = np.array([0, 0, 0, 41, 100, 227, 440, 592, 813, 1139, 1523, 1896,
                      2225, 2609, 2860, 3045, 3269, 3486, 3703, 3902, 4045,
                      4147, 4247, 4309, 4339, 4379, 4400], dtype=float)
n_weeks = len(observed)
t_start, t_end, dt = 0, n_weeks - 1, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
initial_conditions = [11000, 1, 0, 0, 0, 0, 0]

CROSSOVER_VALUES = [0.1, 0.3]
NUM_LHS_SAMPLES = 20000
TOP_K = 5
X0_X1_MIN_GAP = 1.0

PARAM_NAMES_SINGLE = ["b1", "b2", "x0", "c", "gamma1", "gamma2", "mu1", "mu2", "sensitivity", "dispose_rate"]
BOUNDS_SINGLE = [(0.01, 0.2), (5.00, 20.00), (3.5, 15), (0.02, 2.0), (0.35, 0.7),
                 (1.0, 1.4), (1.0, 1.4), (1.0, 10), (0.005, 0.05), (2.0, 7.5)]
FULL_NAMES_SINGLE = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2",
                     "mu1", "mu2", "T_perceive", "sensitivity", "dispose_rate"]

PARAM_NAMES_DOUBLE = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2",
                      "mu1", "mu2", "sensitivity", "dispose_rate"]
BOUNDS_DOUBLE = [(0.01, 0.2), (5.00, 20.00), (3.5, 15), (17, 21), (0.02, 2.0), (0.8, 4.0),
                 (0.35, 0.7), (1.0, 1.4), (1.0, 1.4), (1.0, 10), (0.005, 0.05), (2.0, 7.5)]
IDX_X0_D = PARAM_NAMES_DOUBLE.index("x0")
IDX_X1_D = PARAM_NAMES_DOUBLE.index("x1")


def simulate_default(model_fn, full_params):
    """Default RK45, matches what produced the original crossover result."""
    try:
        sol = solve_ivp(model_fn, [t_start, t_end], initial_conditions,
                         args=(full_params,), t_eval=t_points, method="RK45")
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
    predicted = simulate_default(plague_model_single, full)
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
    predicted = simulate_default(plague_model_double, full)
    if predicted is None:
        return 1e12
    sse = np.sum((observed - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def scale_parameters(params):
    scaling_factors = np.maximum(np.abs(params), 1e-3)
    return params / scaling_factors, scaling_factors


def refit_at_fixed_t_perceive(sse_fn, param_names, bounds, t_perceive_fixed, seed):
    """Same multi-start procedure as malta_T_perceive_sensitivity_check.py,
    but returns the actual best-fit reduced parameter vector, not just SSE."""
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
    best_params = None
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
            best_params = optimized
    return best_sse, best_params


def solver_comparison(model_fn, full_params, label):
    """Re-solve the SAME best-fit parameters under three solver
    configurations and report SSE for each."""
    runs = [
        ("Default RK45 (loose tolerances)", dict(method="RK45")),
        ("RK45, tolerances tightened 10,000x", dict(method="RK45", rtol=1e-10, atol=1e-12)),
        ("Radau (implicit stiff-solver), same tight tolerances", dict(method="Radau", rtol=1e-10, atol=1e-12)),
    ]
    print(f"\n  {label}:")
    sses = []
    for run_label, solver_kwargs in runs:
        try:
            sol = solve_ivp(model_fn, [t_start, t_end], initial_conditions,
                             args=(full_params,), t_eval=t_points, **solver_kwargs)
        except (ValueError, FloatingPointError, OverflowError):
            print(f"    {run_label:<50} SOLVE FAILED")
            sses.append(None)
            continue
        if not sol.success or not np.all(np.isfinite(sol.y)):
            print(f"    {run_label:<50} SOLVE FAILED (not successful / non-finite)")
            sses.append(None)
            continue
        predicted = sol.y[5][::steps_per_week][:n_weeks]
        sse = float(np.sum((observed - predicted) ** 2))
        print(f"    {run_label:<50} SSE = {sse:.3f}")
        sses.append(sse)
    return sses


def main():
    csv_rows = []
    for t_val in CROSSOVER_VALUES:
        print(f"\n{'='*70}\nT_perceive = {t_val}\n{'='*70}")

        s_sse, s_params = refit_at_fixed_t_perceive(sse_single, PARAM_NAMES_SINGLE, BOUNDS_SINGLE, t_val, seed=42)
        d_sse, d_params = refit_at_fixed_t_perceive(sse_double, PARAM_NAMES_DOUBLE, BOUNDS_DOUBLE, t_val, seed=42)
        print(f"Refit confirms: single SSE={s_sse:.2f}, double SSE={d_sse:.2f} "
              f"({'single wins' if s_sse < d_sse else 'double wins'})")

        d_s = dict(zip(PARAM_NAMES_SINGLE, s_params))
        d_s["T_perceive"] = t_val
        full_single = np.array([d_s.get(n, 1.0) for n in FULL_NAMES_SINGLE])

        d_d = dict(zip(PARAM_NAMES_DOUBLE, d_params))
        full_double = np.array([d_d["b1"], d_d["b2"], d_d["x0"], d_d["x1"], d_d["c"], d_d["c1"],
                                 d_d["gamma1"], d_d["gamma2"], d_d["mu1"], d_d["mu2"],
                                 t_val, d_d["sensitivity"], d_d["dispose_rate"]])

        sses_single = solver_comparison(plague_model_single, full_single, "Single-sigmoid")
        sses_double = solver_comparison(plague_model_double, full_double, "Double-sigmoid")

        valid_single = [s for s in sses_single if s is not None]
        valid_double = [s for s in sses_double if s is not None]
        if len(valid_single) >= 2:
            spread = max(valid_single) - min(valid_single)
            stable = spread < 0.01 * min(valid_single)
            print(f"\n  Single-sigmoid SSE spread across solvers: {spread:.3f} "
                  f"({'STABLE' if stable else 'UNSTABLE -- WARNING'})")
            csv_rows.append({"T_perceive": t_val, "model": "single", "spread": spread,
                              "stable": stable})
        if len(valid_double) >= 2:
            spread = max(valid_double) - min(valid_double)
            stable = spread < 0.01 * min(valid_double)
            print(f"  Double-sigmoid SSE spread across solvers: {spread:.3f} "
                  f"({'STABLE' if stable else 'UNSTABLE -- WARNING'})")
            csv_rows.append({"T_perceive": t_val, "model": "double", "spread": spread,
                              "stable": stable})

    pd.DataFrame(csv_rows).to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {OUTPUT_CSV}")

    print(f"\n{'='*70}")
    print("Interpretation:")
    print("- If SSE agrees closely (small spread, 'STABLE') across all three solvers")
    print("  for both models at both crossover points, the crossover is a genuine")
    print("  result -- Malta's T_perceive dependency is real and needs honest framing.")
    print("- If SSE disagrees substantially ('UNSTABLE') for either model, the original")
    print("  default-RK45 result at that point cannot be trusted -- the crossover may")
    print("  be a numerical artifact, not a real finding.")


if __name__ == "__main__":
    main()