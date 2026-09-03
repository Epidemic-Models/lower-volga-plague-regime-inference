"""
malta_double_sigmoidal_dispose_rate_final_confirm.py

Final, full-scale confirmation of the escape-route-corrected sigmoidal
double-sigmoid result for Malta, mirroring
malta_dispose_rate_final_confirm.py (already used successfully for the
constant-gamma,mu model). Fixes dispose_rate at 3.0, the threshold
point identified by the lighter 40k-sample diagnostic (b2 stops
pinning between 2.0 and 3.0), and refits everything else at the full
N=100,000/300,000 two-sample-size comparison.

Required:
    plague_double_sigmoid_model.py in the vetlyanka/ folder.
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "fits", "double")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VETLYANKA_DIR = "/Users/bornalid/PycharmProjects/lower-volga-plague-regime-inference/vetlyanka "
sys.path.insert(0, VETLYANKA_DIR)

from plague_double_sigmoid_model import plague_model

PARAM_NAMES = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2",
               "mu1", "mu2", "T_perceive", "sensitivity"]
BOUNDS = [
    (0.001, 8.0), (1.0, 100.0), (1.0, 20.0), (5.0, 27.0), (0.01, 5.0), (0.01, 5.0),
    (0.35, 0.7), (1.0, 1.4), (1.0, 1.4), (1.0, 10), (0.1, 20.0), (0.0001, 0.5),
]
K = len(BOUNDS)
IDX_X0, IDX_X1 = 2, 3
X0_X1_MIN_GAP = 1.0

DISPOSE_RATE_FIXED = 3.0
SAMPLE_SIZES_TO_COMPARE = [100000, 300000]
LHS_SEED = 42
TOP_K = 10

observed = np.array([0, 0, 0, 41, 100, 227, 440, 592, 813, 1139, 1523, 1896,
                      2225, 2609, 2860, 3045, 3269, 3486, 3703, 3902, 4045,
                      4147, 4247, 4309, 4339, 4379, 4400], dtype=float)
n_weeks = len(observed)
t_start, t_end, dt = 0, n_weeks - 1, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
initial_conditions = [11000, 1, 0, 0, 0, 0, 0]


class TimeoutError_(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError_()


def reduced_to_full(reduced, dispose_rate_fixed):
    b1, b2, x0, x1, c, c1, gamma1, gamma2, mu1, mu2, T_perceive, sensitivity = reduced
    return np.array([b1, b2, x0, x1, c, c1, gamma1, gamma2, mu1, mu2,
                      T_perceive, sensitivity, dispose_rate_fixed])


def simulate_weekly(reduced, dispose_rate_fixed):
    full = reduced_to_full(reduced, dispose_rate_fixed)
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


def sse_unscaled(reduced, dispose_rate_fixed):
    if reduced[IDX_X0] >= reduced[IDX_X1] - X0_X1_MIN_GAP:
        return 1e12
    predicted = simulate_weekly(reduced, dispose_rate_fixed)
    if predicted is None:
        return 1e12
    sse = np.sum((observed - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


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
    print(f"\n{'='*60}\nSigmoidal double, dispose_rate={DISPOSE_RATE_FIXED}, N={num_samples}\n{'='*60}")
    sampler = qmc.LatinHypercube(d=K, seed=LHS_SEED)
    unit = sampler.random(n=num_samples)
    samples = np.array([[row[j] * (BOUNDS[j][1] - BOUNDS[j][0]) + BOUNDS[j][0] for j in range(K)]
                         for row in unit])

    scored = [(sse_unscaled(row, DISPOSE_RATE_FIXED), row) for row in samples]
    scored.sort(key=lambda p: p[0])
    top = scored[:TOP_K]

    best_sse, best_params = np.inf, None
    for lhs_sse, start in top:
        sx0, sf = scale_parameters(start)
        sb = [(lo/f, hi/f) for (lo, hi), f in zip(BOUNDS, sf)]
        result = minimize(lambda sp: sse_unscaled(sp * sf, DISPOSE_RATE_FIXED), sx0,
                           method="L-BFGS-B", bounds=sb,
                           options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})
        opt = result.x * sf
        psse = sse_unscaled(opt, DISPOSE_RATE_FIXED)
        if psse < best_sse:
            best_sse, best_params = psse, opt

    pinned = check_pinned(best_params, PARAM_NAMES, BOUNDS)
    print(f"Best SSE = {best_sse:.6f}")
    print(f"Pinned: {pinned if pinned else 'none'}")
    return best_sse, best_params, pinned


def main():
    results = {}
    for n in SAMPLE_SIZES_TO_COMPARE:
        results[n] = run_fit(n)

    sizes = SAMPLE_SIZES_TO_COMPARE
    pct = abs(results[sizes[0]][0] - results[sizes[-1]][0]) / results[sizes[-1]][0] * 100
    consistent = pct < 1.0 and set(results[sizes[0]][2]) == set(results[sizes[-1]][2])
    print(f"\n{'CONSISTENT' if consistent else 'NOT YET CONSISTENT'}: SSE changed {pct:.2f}%")

    best_sse, best_params, pinned = results[sizes[-1]]
    print(f"\nFinal (N={sizes[-1]}) at dispose_rate={DISPOSE_RATE_FIXED}: SSE={best_sse:.3f}")
    for name, val in zip(PARAM_NAMES, best_params):
        print(f"  {name:15s} = {val:.6f}")

    pd.DataFrame({"Parameter": PARAM_NAMES, "Bound_lo": [b[0] for b in BOUNDS],
                  "Bound_hi": [b[1] for b in BOUNDS], "Best_Fit": best_params}
                 ).to_csv(os.path.join(OUTPUT_DIR, "malta_sigmoidal_dr3_polished_fit.csv"), index=False)
    print(f"\nCompare against single-sigmoid's confirmed SSE=18,961 for the final,")
    print(f"fully-confirmed double-vs-single margin for Malta's sigmoidal pair.")


if __name__ == "__main__":
    main()