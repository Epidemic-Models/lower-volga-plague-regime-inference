"""
malta_double_constant_b2_joint_escape_check.py

Diagnostic: does b2 keep pinning because the box is simply too small, or
because b2 has a joint escape route with another parameter (most likely
sensitivity, since an extreme peak transmission rate can be "hidden"
behind strong enough behavioral suppression once P builds up) -- the
same signature already found and fixed for T_perceive+dispose_rate
earlier this project. If sensitivity (and/or dispose_rate) climbs in
lockstep as b2's ceiling is pushed further out, that confirms a joint
escape, not simple under-bounding, and the fix is the same kind used
before: anchor one of the two independently rather than letting them
chase each other indefinitely.

Runs malta_fit_double_constant.py's exact fitting procedure at several
different b2 ceilings, holding every other bound fixed, and reports
best-fit b2/sensitivity/dispose_rate at each to compare directly.

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
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # malta/, now that this sits in diagnostics/
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VETLYANKA_DIR = "/Users/bornalid/PycharmProjects/lower-volga-plague-regime-inference/vetlyanka "
sys.path.insert(0, VETLYANKA_DIR)

from plague_double_sigmoid_model import plague_model

REDUCED_PARAM_NAMES = ["b1", "b2", "x0", "x1", "c", "c1",
                        "gamma_const", "mu_const", "T_perceive", "sensitivity", "dispose_rate"]
GAMMA_CONST_BOUND = (0.35, 1.4)
MU_CONST_BOUND = (1.0, 10.0)

# Everything except b2 stays fixed at the already-established working bounds.
FIXED_BOUNDS = {
    "b1": (0.001, 8.0), "x0": (1.0, 20.0), "x1": (5.0, 27.0),
    "c": (0.01, 5.0), "c1": (0.01, 5.0),
    "gamma_const": GAMMA_CONST_BOUND, "mu_const": MU_CONST_BOUND,
    "T_perceive": (0.1, 20.0), "sensitivity": (0.0001, 0.5), "dispose_rate": (0.1, 15.0),
}
B2_CEILINGS_TO_TEST = [100.0, 200.0, 500.0, 1000.0]

IDX_X0, IDX_X1 = 2, 3
X0_X1_MIN_GAP = 1.0

observed = np.array([0, 0, 0, 41, 100, 227, 440, 592, 813, 1139, 1523, 1896,
                      2225, 2609, 2860, 3045, 3269, 3486, 3703, 3902, 4045,
                      4147, 4247, 4309, 4339, 4379, 4400], dtype=float)
n_weeks = len(observed)
t_start, t_end, dt = 0, n_weeks - 1, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
initial_conditions = [11000, 1, 0, 0, 0, 0, 0]

NUM_LHS_SAMPLES = 40000  # smaller than the full 100k/300k runs -- this is a
                          # targeted diagnostic across 4 ceilings, not a final fit
TOP_K = 5
LHS_SEED = 42


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


def scale_parameters(params):
    sf = np.maximum(np.abs(params), 1e-3)
    return params / sf, sf


def fit_at_b2_ceiling(b2_ceiling):
    bounds = [
        FIXED_BOUNDS["b1"], (1.0, b2_ceiling), FIXED_BOUNDS["x0"], FIXED_BOUNDS["x1"],
        FIXED_BOUNDS["c"], FIXED_BOUNDS["c1"], FIXED_BOUNDS["gamma_const"], FIXED_BOUNDS["mu_const"],
        FIXED_BOUNDS["T_perceive"], FIXED_BOUNDS["sensitivity"], FIXED_BOUNDS["dispose_rate"],
    ]
    K = len(bounds)

    sampler = qmc.LatinHypercube(d=K, seed=LHS_SEED)
    unit = sampler.random(n=NUM_LHS_SAMPLES)
    samples = np.array([[row[j] * (bounds[j][1] - bounds[j][0]) + bounds[j][0] for j in range(K)]
                         for row in unit])

    scored = [(sse_unscaled(row), row) for row in samples]
    scored.sort(key=lambda p: p[0])
    top = scored[:TOP_K]

    best_sse, best_params = np.inf, None
    for lhs_sse, start in top:
        sx0, sf = scale_parameters(start)
        sb = [(lo/f, hi/f) for (lo, hi), f in zip(bounds, sf)]
        result = minimize(lambda sp: sse_unscaled(sp * sf), sx0, method="L-BFGS-B", bounds=sb,
                           options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})
        opt = result.x * sf
        psse = sse_unscaled(opt)
        if psse < best_sse:
            best_sse, best_params = psse, opt

    return best_sse, best_params


def main():
    print(f"{'b2 ceiling':>12} {'Best SSE':>14} {'best b2':>12} {'best sensitivity':>18} {'best dispose_rate':>18}")
    print("-" * 78)
    results = []
    for ceiling in B2_CEILINGS_TO_TEST:
        sse, params = fit_at_b2_ceiling(ceiling)
        b2_val = params[1]
        sens_val = params[9]
        disp_val = params[10]
        print(f"{ceiling:>12.1f} {sse:>14.3f} {b2_val:>12.3f} {sens_val:>18.6f} {disp_val:>18.6f}")
        results.append({"b2_ceiling": ceiling, "SSE": sse, "best_b2": b2_val,
                         "best_sensitivity": sens_val, "best_dispose_rate": disp_val})

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "malta_b2_joint_escape_check.csv"), index=False)

    print("\nInterpretation:")
    print("- If best_b2 keeps landing at (or very near) each new ceiling, AND SSE keeps")
    print("  improving each time, b2 has no genuine interior optimum in this range at all.")
    print("- If best_sensitivity climbs in step with the b2 ceiling, that confirms a joint")
    print("  escape route (extreme transmission hidden behind extreme suppression) -- the")
    print("  fix is to anchor one of the two independently, not keep widening both.")
    print("- If best_sensitivity stays roughly flat while b2 keeps climbing, the issue is")
    print("  b2 alone wanting to be very large, not a joint escape with sensitivity --")
    print("  check dispose_rate's behavior instead, or look for a different partner.")
    print(f"\nSaved {OUTPUT_DIR}/malta_b2_joint_escape_check.csv")


if __name__ == "__main__":
    main()