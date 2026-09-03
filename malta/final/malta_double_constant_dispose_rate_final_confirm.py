"""
malta_double_constant_dispose_rate_final_confirm.py


Final, full-scale confirmation of the escape-route-corrected double-
sigmoid-constant-gamma,mu result for Malta. Fixes dispose_rate at two
candidate values (2.0, the edge of the confirmed-clean region from
malta_dispose_rate_fixed_check.py; and 7.5, Vetlyanka's own
consistently-preferred value, which gave an anomalous result at only
40,000 samples and needs a properly-resourced recheck) and refits
everything else at the full N=100,000/300,000 two-sample-size
comparison already used for every other confirmed Malta result.

b2 ceiling set to 100 -- confirmed sufficient once dispose_rate is
properly anchored away from the escape-route region (b2 landed at just
18.6 there in the lighter diagnostic, nowhere near 100).

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VETLYANKA_DIR = "/Users/bornalid/PycharmProjects/lower-volga-plague-regime-inference/vetlyanka "
sys.path.insert(0, VETLYANKA_DIR)

from plague_double_sigmoid_model import plague_model

REDUCED_PARAM_NAMES = ["b1", "b2", "x0", "x1", "c", "c1", "gamma_const", "mu_const",
                        "T_perceive", "sensitivity"]
GAMMA_CONST_BOUND = (0.35, 1.4)
MU_CONST_BOUND = (1.0, 10.0)
BOUNDS = [
    (0.001, 8.0),    # b1
    (1.0, 100.0),    # b2
    (1.0, 20.0),     # x0
    (5.0, 27.0),     # x1
    (0.01, 5.0),     # c
    (0.01, 5.0),     # c1
    GAMMA_CONST_BOUND,
    MU_CONST_BOUND,
    (0.1, 20.0),     # T_perceive
    (0.0001, 0.5),   # sensitivity
]
K = len(BOUNDS)
IDX_X0, IDX_X1 = 2, 3
X0_X1_MIN_GAP = 1.0

DISPOSE_RATE_VALUES = [2.0, 7.5]
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
    b1, b2, x0, x1, c, c1, gamma_const, mu_const, T_perceive, sensitivity = reduced
    return np.array([b1, b2, x0, x1, c, c1, gamma_const, gamma_const,
                      mu_const, mu_const, T_perceive, sensitivity, dispose_rate_fixed])


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


def run_fit(num_samples, dispose_rate_fixed):
    print(f"\n{'='*60}\ndispose_rate={dispose_rate_fixed}, N={num_samples}\n{'='*60}")
    sampler = qmc.LatinHypercube(d=K, seed=LHS_SEED)
    unit = sampler.random(n=num_samples)
    samples = np.array([[row[j] * (BOUNDS[j][1] - BOUNDS[j][0]) + BOUNDS[j][0] for j in range(K)]
                         for row in unit])

    scored = [(sse_unscaled(row, dispose_rate_fixed), row) for row in samples]
    scored.sort(key=lambda p: p[0])
    top = scored[:TOP_K]

    best_sse, best_params = np.inf, None
    for lhs_sse, start in top:
        sx0, sf = scale_parameters(start)
        sb = [(lo/f, hi/f) for (lo, hi), f in zip(BOUNDS, sf)]
        result = minimize(lambda sp: sse_unscaled(sp * sf, dispose_rate_fixed), sx0,
                           method="L-BFGS-B", bounds=sb,
                           options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})
        opt = result.x * sf
        psse = sse_unscaled(opt, dispose_rate_fixed)
        if psse < best_sse:
            best_sse, best_params = psse, opt

    pinned = check_pinned(best_params, REDUCED_PARAM_NAMES, BOUNDS)
    print(f"Best SSE = {best_sse:.6f}")
    print(f"Pinned: {pinned if pinned else 'none'}")
    return best_sse, best_params, pinned


def compute_peak_suppression(best_params, dispose_rate_fixed):
    """Peak behavioral suppression (%), computed from the confirmed
    best-fit parameters using the same rho_beta = (1-exp(-lambda*P))*100
    formula as Vetlyanka's own behavioural_adaptation_figure.py, so this
    number is directly comparable to Vetlyanka's confirmed figures."""
    full_params = reduced_to_full(best_params, dispose_rate_fixed)
    sol = solve_ivp(plague_model, [t_start, t_end], initial_conditions,
                     args=(full_params,), t_eval=t_points, method="RK45")
    P = sol.y[6]
    sensitivity = full_params[11]
    rho_beta = (1 - np.exp(-sensitivity * P)) * 100
    peak_idx = np.argmax(rho_beta)
    return float(rho_beta[peak_idx]), float(t_points[peak_idx])


def main():
    all_results = {}
    for dr in DISPOSE_RATE_VALUES:
        results = {}
        for n in SAMPLE_SIZES_TO_COMPARE:
            results[n] = run_fit(n, dr)

        sizes = SAMPLE_SIZES_TO_COMPARE
        pct = abs(results[sizes[0]][0] - results[sizes[-1]][0]) / results[sizes[-1]][0] * 100
        consistent = pct < 1.0 and set(results[sizes[0]][2]) == set(results[sizes[-1]][2])
        print(f"\n{'CONSISTENT' if consistent else 'NOT YET CONSISTENT'} at dispose_rate={dr}: SSE changed {pct:.2f}%")

        best_sse, best_params, pinned = results[sizes[-1]]
        print(f"Final (N={sizes[-1]}) at dispose_rate={dr}: SSE={best_sse:.3f}")
        for name, val in zip(REDUCED_PARAM_NAMES, best_params):
            print(f"  {name:15s} = {val:.6f}")

        peak_pct, peak_week = compute_peak_suppression(best_params, dr)
        print(f"  Peak behavioral suppression: {peak_pct:.1f}% at week {peak_week:.1f}")

        all_results[dr] = {"SSE": best_sse, "b2": best_params[1], "pinned": pinned,
                            "consistent": consistent, "peak_suppression_pct": peak_pct,
                            "peak_suppression_week": peak_week}

        pd.DataFrame({"Parameter": REDUCED_PARAM_NAMES, "Bound_lo": [b[0] for b in BOUNDS],
                      "Bound_hi": [b[1] for b in BOUNDS], "Best_Fit": best_params}
                     ).to_csv(os.path.join(OUTPUT_DIR, f"malta_double_constant_dr{dr}_polished_fit.csv"), index=False)

    print(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")
    print(f"{'dispose_rate':>14} {'SSE':>12} {'b2':>10} {'consistent?':>12} {'peak supp.':>12} {'pinned':>30}")
    for dr, r in all_results.items():
        print(f"{dr:>14.2f} {r['SSE']:>12.3f} {r['b2']:>10.3f} {str(r['consistent']):>12} "
              f"{r['peak_suppression_pct']:>11.1f}% {str(r['pinned']):>30}")

    print("\nCompare against single-constant's confirmed SSE=13,549 to get the final,")
    print("fully-confirmed double-vs-single margin for Malta's constant-gamma,mu pair.")


if __name__ == "__main__":
    main()