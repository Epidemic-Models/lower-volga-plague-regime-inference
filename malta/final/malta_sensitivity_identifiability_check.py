"""
malta_sensitivity_identifiability_check.py

Tests whether sensitivity (lambda_P) is genuinely identified by Malta's
data, or degenerate with b1/b2 (transmission amplitude) -- a different
question from the already-resolved b2/dispose_rate escape route.

Fixes sensitivity at half, exactly, and double the confirmed final
value (0.004729), refitting everything else (b1, b2, x0, x1, c, c1,
T_perceive; gamma/mu within their historically-grounded bounds;
dispose_rate fixed at 3.0, already resolved) at each. If b1/b2 swing to
compensate and SSE stays close to the confirmed 14,215.55, that
confirms degeneracy -- real suppression effect, but the specific
79.3%/~4x-latent-capacity magnitude is not well-identified. If SSE gets
meaningfully worse away from the confirmed value, sensitivity is
genuinely pinned down by the data.

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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VETLYANKA_DIR = "/Users/bornalid/PycharmProjects/lower-volga-plague-regime-inference/vetlyanka "
sys.path.insert(0, VETLYANKA_DIR)

from plague_double_sigmoid_model import plague_model

PARAM_NAMES = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2", "mu1", "mu2", "T_perceive"]
BOUNDS = [
    (0.001, 8.0), (1.0, 100.0), (1.0, 20.0), (5.0, 27.0), (0.01, 5.0), (0.01, 5.0),
    (0.35, 0.7), (1.0, 1.4), (1.0, 1.4), (1.0, 10), (0.1, 20.0),
]
K = len(BOUNDS)
IDX_X0, IDX_X1 = 2, 3
X0_X1_MIN_GAP = 1.0
DISPOSE_RATE_FIXED = 3.0  # already resolved

CONFIRMED_SENSITIVITY = 0.004729
SENSITIVITY_VALUES_TO_TEST = [CONFIRMED_SENSITIVITY / 2, CONFIRMED_SENSITIVITY, CONFIRMED_SENSITIVITY * 2]
CONFIRMED_SSE = 14215.55

observed = np.array([0, 0, 0, 41, 100, 227, 440, 592, 813, 1139, 1523, 1896,
                      2225, 2609, 2860, 3045, 3269, 3486, 3703, 3902, 4045,
                      4147, 4247, 4309, 4339, 4379, 4400], dtype=float)
n_weeks = len(observed)
t_start, t_end, dt = 0, n_weeks - 1, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
initial_conditions = [11000, 1, 0, 0, 0, 0, 0]

NUM_LHS_SAMPLES = 40000
TOP_K = 8
LHS_SEED = 42


class TimeoutError_(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError_()


def reduced_to_full(reduced, sensitivity_fixed):
    b1, b2, x0, x1, c, c1, gamma1, gamma2, mu1, mu2, T_perceive = reduced
    return np.array([b1, b2, x0, x1, c, c1, gamma1, gamma2, mu1, mu2,
                      T_perceive, sensitivity_fixed, DISPOSE_RATE_FIXED])


def simulate_weekly(reduced, sensitivity_fixed):
    full = reduced_to_full(reduced, sensitivity_fixed)
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
    predicted = sol.y[4][::steps_per_week][:n_weeks]
    if predicted.shape[0] != n_weeks or not np.all(np.isfinite(predicted)):
        return None
    return predicted


def sse_unscaled(reduced, sensitivity_fixed):
    if reduced[IDX_X0] >= reduced[IDX_X1] - X0_X1_MIN_GAP:
        return 1e12
    predicted = simulate_weekly(reduced, sensitivity_fixed)
    if predicted is None:
        return 1e12
    sse = np.sum((observed - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def scale_parameters(params):
    sf = np.maximum(np.abs(params), 1e-3)
    return params / sf, sf


def fit_at_sensitivity(sensitivity_fixed):
    sampler = qmc.LatinHypercube(d=K, seed=LHS_SEED)
    unit = sampler.random(n=NUM_LHS_SAMPLES)
    samples = np.array([[row[j] * (BOUNDS[j][1] - BOUNDS[j][0]) + BOUNDS[j][0] for j in range(K)]
                         for row in unit])

    scored = [(sse_unscaled(row, sensitivity_fixed), row) for row in samples]
    scored.sort(key=lambda p: p[0])
    top = scored[:TOP_K]

    best_sse, best_params = np.inf, None
    for lhs_sse, start in top:
        sx0, sf = scale_parameters(start)
        sb = [(lo/f, hi/f) for (lo, hi), f in zip(BOUNDS, sf)]
        result = minimize(lambda sp: sse_unscaled(sp * sf, sensitivity_fixed), sx0,
                           method="L-BFGS-B", bounds=sb,
                           options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})
        opt = result.x * sf
        psse = sse_unscaled(opt, sensitivity_fixed)
        if psse < best_sse:
            best_sse, best_params = psse, opt

    return best_sse, best_params


def main():
    print(f"Confirmed baseline: sensitivity={CONFIRMED_SENSITIVITY}, SSE={CONFIRMED_SSE}\n")
    print(f"{'sensitivity':>14} {'SSE':>12} {'%worse than baseline':>22} {'b1':>8} {'b2':>8}")
    print("-" * 70)
    results = []
    for sens in SENSITIVITY_VALUES_TO_TEST:
        sse, params = fit_at_sensitivity(sens)
        pct_worse = (sse - CONFIRMED_SSE) / CONFIRMED_SSE * 100
        b1, b2 = params[0], params[1]
        print(f"{sens:>14.6f} {sse:>12.2f} {pct_worse:>21.1f}% {b1:>8.3f} {b2:>8.3f}")
        results.append({"sensitivity_fixed": sens, "SSE": sse, "pct_worse_than_baseline": pct_worse,
                         "b1": b1, "b2": b2})

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "malta_sensitivity_identifiability_check.csv"), index=False)

    print("\nInterpretation:")
    print("- If SSE stays close to baseline (small % worse) at BOTH half and double sensitivity,")
    print("  while b1/b2 shift meaningfully to compensate, that confirms degeneracy: real")
    print("  suppression effect, but the specific 79.3% magnitude is not well-identified.")
    print("- If SSE gets substantially worse away from the confirmed value, sensitivity is")
    print("  genuinely pinned down by the data -- the 79.3% figure can be reported as-is.")
    print(f"\nSaved {OUTPUT_DIR}/malta_sensitivity_identifiability_check.csv")


if __name__ == "__main__":
    main()