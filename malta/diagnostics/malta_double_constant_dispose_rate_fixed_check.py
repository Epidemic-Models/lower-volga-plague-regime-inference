"""
malta_double_constant_dispose_rate_fixed_check.py

Follow-up to malta_b2_joint_escape_check.py: dispose_rate sat at exactly
0.1 (its floor) across every b2 ceiling tested, while b2 kept pinning
and SSE kept improving with no sign of leveling off, at full search
power -- a genuine joint escape route (unboundedly high peak
transmission + unboundedly slow corpse disposal jointly faking an
ever-better fit through lingering infectious corpses), not simple
under-bounding.

This script closes the escape route the same way T_perceive/dispose_rate
was resolved for Vetlyanka: fix dispose_rate at several independently-
chosen values (not searched) and refit b2 (moderate ceiling) at each --
if b2 THEN converges to a genuine interior value instead of pinning,
that confirms the diagnosis and identifies which dispose_rate value
lets the model behave sensibly.

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

# Reduced space with dispose_rate REMOVED (fixed per test, not searched)
REDUCED_PARAM_NAMES = ["b1", "b2", "x0", "x1", "c", "c1", "gamma_const", "mu_const",
                        "T_perceive", "sensitivity"]
GAMMA_CONST_BOUND = (0.35, 1.4)
MU_CONST_BOUND = (1.0, 10.0)
B2_CEILING = 100.0  # moderate -- if the escape route is really closed, b2 shouldn't need more than this
BOUNDS = [
    (0.001, 8.0),    # b1
    (1.0, B2_CEILING),  # b2
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

DISPOSE_RATE_VALUES_TO_TEST = [0.1, 0.5, 1.0, 2.0, 5.0, 7.5]  # 7.5 matches Vetlyanka's own consistently-pinned value

observed = np.array([0, 0, 0, 41, 100, 227, 440, 592, 813, 1139, 1523, 1896,
                      2225, 2609, 2860, 3045, 3269, 3486, 3703, 3902, 4045,
                      4147, 4247, 4309, 4339, 4379, 4400], dtype=float)
n_weeks = len(observed)
t_start, t_end, dt = 0, n_weeks - 1, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
initial_conditions = [11000, 1, 0, 0, 0, 0, 0]

NUM_LHS_SAMPLES = 40000
TOP_K = 5
LHS_SEED = 42


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


def fit_at_dispose_rate(dispose_rate_fixed):
    sampler = qmc.LatinHypercube(d=K, seed=LHS_SEED)
    unit = sampler.random(n=NUM_LHS_SAMPLES)
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

    return best_sse, best_params


def main():
    print(f"b2 ceiling fixed at {B2_CEILING}; testing dispose_rate fixed at several values\n")
    print(f"{'dispose_rate':>14} {'Best SSE':>14} {'best b2':>12} {'b2 pinned?':>12}")
    print("-" * 56)
    results = []
    for dr in DISPOSE_RATE_VALUES_TO_TEST:
        sse, params = fit_at_dispose_rate(dr)
        b2_val = params[1]
        pinned = "YES" if b2_val > B2_CEILING * 0.95 else "no"
        print(f"{dr:>14.2f} {sse:>14.3f} {b2_val:>12.3f} {pinned:>12}")
        results.append({"dispose_rate_fixed": dr, "SSE": sse, "best_b2": b2_val, "b2_pinned": pinned})

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "malta_dispose_rate_fixed_check.csv"), index=False)

    print("\nInterpretation:")
    print("- If b2 stops pinning (lands well below the ceiling) once dispose_rate is fixed")
    print("  at some value, that confirms the joint escape and identifies which dispose_rate")
    print("  value closes it -- use that region for a defensible constant dispose_rate.")
    print("- If b2 keeps pinning regardless of the fixed dispose_rate value, the escape")
    print("  route involves a different parameter, or b2 genuinely wants to be very large")
    print("  for a reason not yet identified.")
    print(f"\nSaved {OUTPUT_DIR}/malta_dispose_rate_fixed_check.csv")


if __name__ == "__main__":
    main()