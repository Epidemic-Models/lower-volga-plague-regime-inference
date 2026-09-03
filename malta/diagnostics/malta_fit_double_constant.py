"""
malta_fit_double_constant.py

Constant-gamma,mu ablation for Malta's double-sigmoid model. Fits
against DDI (plague-only cumulative deaths).

b2 widened this round -- was hard-pinning at 50.0.
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostics")

VETLYANKA_DIR = "/Users/bornalid/PycharmProjects/lower-volga-plague-regime-inference/vetlyanka "
sys.path.insert(0, VETLYANKA_DIR)

from plague_double_sigmoid_model import plague_model

REDUCED_PARAM_NAMES = ["b1", "b2", "x0", "x1", "c", "c1",
                        "gamma_const", "mu_const", "T_perceive", "sensitivity", "dispose_rate"]
GAMMA_CONST_BOUND = (0.35, 1.4)
MU_CONST_BOUND = (1.0, 10.0)
REDUCED_BOUNDS = [
    (0.001, 8.0),    # b1
    (1.0, 300.0),    # b2 -- widened, was hard-pinning at 50.0
    (1.0, 20.0),     # x0
    (5.0, 27.0),     # x1
    (0.01, 5.0),     # c
    (0.01, 5.0),     # c1
    GAMMA_CONST_BOUND,
    MU_CONST_BOUND,
    (0.1, 20.0),     # T_perceive
    (0.0001, 0.5),   # sensitivity
    (0.1, 15.0),     # dispose_rate
]
K = len(REDUCED_PARAM_NAMES)
IDX_X0, IDX_X1 = 2, 3
X0_X1_MIN_GAP = 1.0

FULL_PARAM_NAMES = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2",
                     "mu1", "mu2", "T_perceive", "sensitivity", "dispose_rate"]

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


def reduced_to_full(reduced):
    b1, b2, x0, x1, c, c1, gamma_const, mu_const, T_perceive, sensitivity, dispose_rate = reduced
    return np.array([b1, b2, x0, x1, c, c1, gamma_const, gamma_const,
                      mu_const, mu_const, T_perceive, sensitivity, dispose_rate])


def simulate_weekly(reduced):
    full = reduced_to_full(reduced)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(5)
    try:
        sol = solve_ivp(plague_model, [t_start, t_end], initial_conditions,
                         args=(full,), t_eval=t_points, method="RK45",
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


def sse_unscaled(reduced):
    if reduced[IDX_X0] >= reduced[IDX_X1] - X0_X1_MIN_GAP:
        return 1e12
    predicted = simulate_weekly(reduced)
    if predicted is None:
        return 1e12
    sse = np.sum((observed - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def sse_objective(scaled, sf):
    return sse_unscaled(scaled * sf)


def scale_parameters(params):
    sf = np.maximum(np.abs(params), 1e-3)
    return params / sf, sf


def check_pinned(params, names, bounds, tol=0.05):
    pinned = []
    for name, val, (lo, hi) in zip(names, params, bounds):
        span = hi - lo
        if val < lo + tol * span or val > hi - tol * span:
            wall = "lower" if val < lo + tol * span else "upper"
            pinned.append(f"{name}({wall})")
    return pinned


def run_fit(num_samples):
    print(f"\n{'='*60}\nMalta double-sigmoid, CONSTANT gamma,mu, N={num_samples}\n{'='*60}")
    sampler = qmc.LatinHypercube(d=K, seed=LHS_SEED)
    unit = sampler.random(n=num_samples)
    samples = np.array([[row[j] * (REDUCED_BOUNDS[j][1] - REDUCED_BOUNDS[j][0]) + REDUCED_BOUNDS[j][0]
                          for j in range(K)] for row in unit])

    scored = [(sse_unscaled(row), row) for row in samples]
    scored.sort(key=lambda p: p[0])
    top = scored[:TOP_K]

    best_sse, best_params = np.inf, None
    multistart_log = []
    for rank, (lhs_sse, start) in enumerate(top):
        sx0, sf = scale_parameters(start)
        sb = [(lo/f, hi/f) for (lo, hi), f in zip(REDUCED_BOUNDS, sf)]
        result = minimize(sse_objective, sx0, args=(sf,), method="L-BFGS-B", bounds=sb,
                           options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})
        opt = result.x * sf
        psse = sse_unscaled(opt)
        multistart_log.append({"rank": rank+1, "lhs_sse": lhs_sse, "polished_sse": psse})
        if psse < best_sse:
            best_sse, best_params = psse, opt

    pinned = check_pinned(best_params, REDUCED_PARAM_NAMES, REDUCED_BOUNDS)
    print(f"Best SSE = {best_sse:.6f}")
    print(f"Pinned: {pinned if pinned else 'none'}")
    return best_sse, best_params, pinned, multistart_log


def main():
    results = {}
    for n in SAMPLE_SIZES_TO_COMPARE:
        results[n] = run_fit(n)

    print(f"\n{'='*60}\nCOMPARISON ACROSS SAMPLE SIZES\n{'='*60}")
    for n, (sse, _, pinned, _) in results.items():
        print(f"  N={n:>7}: SSE={sse:.3f}, pinned={pinned if pinned else 'none'}")

    sizes = SAMPLE_SIZES_TO_COMPARE
    pct = abs(results[sizes[0]][0] - results[sizes[-1]][0]) / results[sizes[-1]][0] * 100
    consistent = pct < 1.0 and set(results[sizes[0]][2]) == set(results[sizes[-1]][2])
    print(f"\n{'CONSISTENT' if consistent else 'NOT YET CONSISTENT'}: SSE changed {pct:.2f}%")

    best_sse, best_params, pinned, multistart_log = results[sizes[-1]]
    print(f"\nBest-fit parameters (N={sizes[-1]}):")
    for name, val in zip(REDUCED_PARAM_NAMES, best_params):
        print(f"  {name:15s} = {val:.6f}")

    pd.DataFrame(multistart_log).to_csv(os.path.join(OUTPUT_DIR, "malta_double_constant_multistart_log.csv"), index=False)
    pd.DataFrame({"Parameter": REDUCED_PARAM_NAMES, "Bound_lo": [b[0] for b in REDUCED_BOUNDS],
                  "Bound_hi": [b[1] for b in REDUCED_BOUNDS], "Best_Fit": best_params}
                 ).to_csv(os.path.join(OUTPUT_DIR, "malta_double_constant_polished_fit.csv"), index=False)
    print(f"\nSaved {OUTPUT_DIR}/malta_double_constant_polished_fit.csv")


if __name__ == "__main__":
    main()