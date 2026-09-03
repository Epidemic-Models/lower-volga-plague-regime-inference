"""
vetlyanka_full_correlation_screen.py

Same correlation screen already run for Malta's double-sigmoid fit,
applied to Vetlyanka's own confirmed double-sigmoid fit -- this time
using the REAL, authoritative bounds from vetlyanka_bounds.py, not an
approximation.

Purpose: give a real, substantive answer to Reviewer #3's question
about how correlated parameter estimates were handled in practice --
not just that a top-quality ensemble was retained, but what that
ensemble actually shows about pairwise parameter dependence.

Required:
    plague_double_sigmoid_model.py in the same folder or on the path.
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

from plague_double_sigmoid_model import plague_model

# Real, authoritative bounds -- from vetlyanka_bounds.py, PARAMETERS list.
PARAM_NAMES = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2",
               "mu1", "mu2", "T_perceive", "sensitivity", "dispose_rate"]
BOUNDS = [
    (0.001, 0.20),   # b1
    (5.00, 25.00),   # b2
    (5, 20),         # x0
    (14, 20),        # x1
    (0.02, 2.0),     # c
    (0.8, 4.0),      # c1
    (0.35, 0.7),     # gamma1
    (1.167, 1.4),    # gamma2
    (1.167, 1.4),    # mu1
    (1.75, 14),      # mu2
    (0.5, 20.0),     # T_perceive
    (0.005, 0.2),    # sensitivity
    (2.0, 7.5),      # dispose_rate
]
K = len(BOUNDS)
IDX_X0, IDX_X1 = 2, 3
X0_X1_MIN_GAP = 1.0

QUALITY_THRESHOLD = 1.20  # keep every candidate within 20% of best SSE found
NUM_LHS_SAMPLES = 60000
TOP_K_TO_POLISH = 30
LHS_SEED = 42

# Vetlyanka's real weekly cumulative death data (all-cause, DR)
observed = np.array([3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19, 27, 34, 90, 259,
                      313, 345, 364, 376, 376, 376, 376], dtype=float)
n_weeks = len(observed)
t_start, t_end, dt = 0, n_weeks - 1, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]


class TimeoutError_(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError_()


def simulate_weekly(params):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(5)
    try:
        sol = solve_ivp(plague_model, [t_start, t_end], initial_conditions,
                         args=(params,), t_eval=t_points, method="RK45",
                         rtol=1e-6, atol=1e-8)
    except (ValueError, FloatingPointError, OverflowError, TimeoutError_):
        return None
    finally:
        signal.alarm(0)
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    predicted = sol.y[5][::steps_per_week][:n_weeks]  # DR, all-cause -- correct for Vetlyanka
    if predicted.shape[0] != n_weeks or not np.all(np.isfinite(predicted)):
        return None
    return predicted


def sse_unscaled(params):
    if params[IDX_X0] >= params[IDX_X1] - X0_X1_MIN_GAP:
        return 1e12
    predicted = simulate_weekly(params)
    if predicted is None:
        return 1e12
    sse = np.sum((observed - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def scale_parameters(params):
    sf = np.maximum(np.abs(params), 1e-3)
    return params / sf, sf


def main():
    print(f"Drawing {NUM_LHS_SAMPLES} LHS samples, polishing top {TOP_K_TO_POLISH}...")
    sampler = qmc.LatinHypercube(d=K, seed=LHS_SEED)
    unit = sampler.random(n=NUM_LHS_SAMPLES)
    samples = np.array([[row[j] * (BOUNDS[j][1] - BOUNDS[j][0]) + BOUNDS[j][0] for j in range(K)]
                         for row in unit])

    scored = [(sse_unscaled(row), row) for row in samples]
    scored.sort(key=lambda p: p[0])
    top = scored[:TOP_K_TO_POLISH]

    polished = []
    for lhs_sse, start in top:
        sx0, sf = scale_parameters(start)
        sb = [(lo/f, hi/f) for (lo, hi), f in zip(BOUNDS, sf)]
        result = minimize(lambda sp: sse_unscaled(sp * sf), sx0, method="L-BFGS-B", bounds=sb,
                           options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})
        opt = result.x * sf
        psse = sse_unscaled(opt)
        polished.append((psse, opt))

    best_sse = min(p[0] for p in polished)
    near_optimal = [p for p in polished if p[0] <= best_sse * QUALITY_THRESHOLD]
    print(f"Best SSE found: {best_sse:.3f} (confirmed baseline: 393.136)")
    print(f"Near-optimal pool (within {int((QUALITY_THRESHOLD-1)*100)}%): {len(near_optimal)} candidates")

    df = pd.DataFrame([p[1] for p in near_optimal], columns=PARAM_NAMES)
    df["SSE"] = [p[0] for p in near_optimal]
    df.to_csv(os.path.join(OUTPUT_DIR, "vetlyanka_near_optimal_pool.csv"), index=False)

    corr = df[PARAM_NAMES].corr()
    corr.to_csv(os.path.join(OUTPUT_DIR, "vetlyanka_parameter_correlation_matrix.csv"))

    print("\nStrong correlations (|r| > 0.6):")
    found_any = False
    for i, p1 in enumerate(PARAM_NAMES):
        for p2 in PARAM_NAMES[i+1:]:
            r = corr.loc[p1, p2]
            if abs(r) > 0.6:
                print(f"  {p1} <-> {p2}: r = {r:.2f}")
                found_any = True
    if not found_any:
        print("  None found.")

    print("\n" + "="*60)
    print("x1 (decline timing) correlations -- the parameter the paper's claim depends on:")
    print("="*60)
    x1_correlations = corr.loc["x1"].drop("x1").sort_values(key=abs, ascending=False)
    print(x1_correlations.round(3))

    print(f"\nSaved {OUTPUT_DIR}/vetlyanka_near_optimal_pool.csv")
    print(f"Saved {OUTPUT_DIR}/vetlyanka_parameter_correlation_matrix.csv")


if __name__ == "__main__":
    main()