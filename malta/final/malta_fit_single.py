"""
malta_fit_single.py

Single-sigmoid fit for the Malta plague outbreak, for direct comparison
against malta_fit_double.py. Fits against DDI (plague-only cumulative
deaths), not DR (all-cause) -- Malta's historical record is plague-
specific, unlike Vetlyanka's all-cause burial register.

Bounds widened across several rounds this session; b1/c widened once
more here to match what already resolved cleanly for
malta_fit_single_constant.py (b1 was hard-pinning at 2.0, c at 5.0).

Required:
    plague_single_sigmoid_model.py in the vetlyanka/ folder.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import qmc

import os
import sys
import signal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "fits", "single")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VETLYANKA_DIR = "/Users/bornalid/PycharmProjects/lower-volga-plague-regime-inference/vetlyanka "
sys.path.insert(0, VETLYANKA_DIR)

from plague_single_sigmoid_model import plague_model

PARAM_NAMES_SINGLE = ["b1", "b2", "x0", "c", "gamma1", "gamma2", "mu1", "mu2",
                      "T_perceive", "sensitivity", "dispose_rate"]
BOUNDS_SINGLE = [
    (0.001, 8.0),    # b1 -- widened again, was hard-pinning at 2.0
    (1.0, 50.0),     # b2
    (1.0, 20.0),     # x0
    (0.001, 8.0),    # c -- widened again, was hard-pinning at 5.0
    (0.35, 0.7),     # gamma1
    (1.0, 1.4),      # gamma2
    (1.0, 1.4),      # mu1
    (1.0, 10),       # mu2
    (0.01, 20.0),    # T_perceive
    (0.0001, 0.5),   # sensitivity
    (0.1, 15.0),     # dispose_rate
]
K_SINGLE = len(PARAM_NAMES_SINGLE)

FULL_PARAM_NAMES = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2",
                     "mu1", "mu2", "T_perceive", "sensitivity", "dispose_rate"]
X1_C1_PLACEHOLDER = 1.0

observed = np.array([0, 0, 0, 41, 100, 227, 440, 592, 813, 1139, 1523, 1896,
                      2225, 2609, 2860, 3045, 3269, 3486, 3703, 3902, 4045,
                      4147, 4247, 4309, 4339, 4379, 4400], dtype=float)
n_weeks = len(observed)

t_start, t_end, dt = 0, n_weeks - 1, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
initial_conditions = [11000, 1, 0, 0, 0, 0, 0]

SAMPLE_SIZES_TO_COMPARE = [100000, 300000]
LHS_SEED = 42
TOP_K = 10


class TimeoutError_(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError_()


def reduced_to_full(reduced_params):
    param_dict = dict(zip(PARAM_NAMES_SINGLE, reduced_params))
    return np.array([param_dict.get(name, X1_C1_PLACEHOLDER) for name in FULL_PARAM_NAMES])


def simulate_weekly(reduced_params):
    full_params = reduced_to_full(reduced_params)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(5)
    try:
        sol = solve_ivp(plague_model, [t_start, t_end], initial_conditions,
                         args=(full_params,), t_eval=t_points, method="RK45",
                         rtol=1e-6, atol=1e-8)
    except (ValueError, FloatingPointError, OverflowError, TimeoutError_):
        return None
    finally:
        signal.alarm(0)
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    predicted = sol.y[4][::steps_per_week][:n_weeks]  # DDI, plague-only
    if predicted.shape[0] != n_weeks or not np.all(np.isfinite(predicted)):
        return None
    return predicted


def sse_unscaled(reduced_params):
    predicted = simulate_weekly(reduced_params)
    if predicted is None:
        return 1e12
    sse = np.sum((observed - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def sse_objective(scaled_params, scaling_factors):
    return sse_unscaled(scaled_params * scaling_factors)


def scale_parameters(params):
    scaling_factors = np.maximum(np.abs(params), 1e-3)
    return params / scaling_factors, scaling_factors


def check_pinned(params, names, bounds, tol=0.05):
    pinned = []
    for name, val, (lo, hi) in zip(names, params, bounds):
        span = hi - lo
        if val < lo + tol * span or val > hi - tol * span:
            wall = "lower" if val < lo + tol * span else "upper"
            pinned.append(f"{name}({wall})")
    return pinned


def run_fit(num_samples):
    print(f"\n{'='*60}\nMalta single-sigmoid fit, NUM_LHS_SAMPLES={num_samples}\n{'='*60}")

    sampler = qmc.LatinHypercube(d=K_SINGLE, seed=LHS_SEED)
    unit = sampler.random(n=num_samples)
    lhs_samples = np.array([
        [row[j] * (BOUNDS_SINGLE[j][1] - BOUNDS_SINGLE[j][0]) + BOUNDS_SINGLE[j][0] for j in range(K_SINGLE)]
        for row in unit
    ])

    scored = [(sse_unscaled(row), row) for row in lhs_samples]
    scored.sort(key=lambda pair: pair[0])
    top_candidates = scored[:TOP_K]

    best_sse = np.inf
    best_params = None
    multistart_log = []

    for rank, (lhs_sse, start_row) in enumerate(top_candidates):
        scaled_x0, scaling_factors = scale_parameters(start_row)
        scaled_bounds = [(lo / sf, hi / sf) for (lo, hi), sf in zip(BOUNDS_SINGLE, scaling_factors)]

        result = minimize(sse_objective, scaled_x0, args=(scaling_factors,),
                           method="L-BFGS-B", bounds=scaled_bounds,
                           options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})

        optimized = result.x * scaling_factors
        polished_sse = sse_unscaled(optimized)
        multistart_log.append({"rank": rank + 1, "lhs_sse": lhs_sse, "polished_sse": polished_sse,
                                "T_perceive": optimized[PARAM_NAMES_SINGLE.index("T_perceive")]})

        if polished_sse < best_sse:
            best_sse = polished_sse
            best_params = optimized

    pinned = check_pinned(best_params, PARAM_NAMES_SINGLE, BOUNDS_SINGLE)
    print(f"Best SSE = {best_sse:.6f}")
    print(f"Pinned parameters: {pinned if pinned else 'none'}")
    return best_sse, best_params, pinned, multistart_log


def main():
    results = {}
    for num_samples in SAMPLE_SIZES_TO_COMPARE:
        results[num_samples] = run_fit(num_samples)

    print(f"\n{'='*60}\nCOMPARISON ACROSS SAMPLE SIZES\n{'='*60}")
    for num_samples, (best_sse, _, pinned, _) in results.items():
        print(f"  N={num_samples:>7}: SSE={best_sse:.3f}, pinned={pinned if pinned else 'none'}")

    sizes = SAMPLE_SIZES_TO_COMPARE
    sse_smaller, sse_larger = results[sizes[0]][0], results[sizes[-1]][0]
    pct_change = abs(sse_smaller - sse_larger) / sse_larger * 100
    consistent = pct_change < 1.0 and set(results[sizes[0]][2]) == set(results[sizes[-1]][2])

    if consistent:
        print(f"\nCONSISTENT: SSE changed only {pct_change:.2f}% and pinned-parameter set is")
        print(f"identical between N={sizes[0]} and N={sizes[-1]} -- N={sizes[0]} was already enough.")
    else:
        print(f"\nNOT YET CONSISTENT: SSE changed {pct_change:.2f}%, or the pinned-parameter set")
        print(f"differs between sample sizes -- N={sizes[0]} was NOT enough.")

    best_sse, best_params, pinned, multistart_log = results[sizes[-1]]
    print(f"\nBest-fit parameters (N={sizes[-1]}, the saved result):")
    for name, val in zip(PARAM_NAMES_SINGLE, best_params):
        print(f"  {name:15s} = {val:.6f}")

    pd.DataFrame(multistart_log).to_csv(os.path.join(OUTPUT_DIR, "malta_single_multistart_log.csv"), index=False)
    pd.DataFrame({"Parameter": PARAM_NAMES_SINGLE,
                  "Bound_lo": [b[0] for b in BOUNDS_SINGLE],
                  "Bound_hi": [b[1] for b in BOUNDS_SINGLE],
                  "Best_Fit": best_params}).to_csv(os.path.join(OUTPUT_DIR, "malta_single_polished_fit.csv"), index=False)
    print(f"\nSaved {OUTPUT_DIR}/malta_single_multistart_log.csv (N={sizes[-1]} run)")
    print(f"Saved {OUTPUT_DIR}/malta_single_polished_fit.csv (N={sizes[-1]} run)")


if __name__ == "__main__":
    main()