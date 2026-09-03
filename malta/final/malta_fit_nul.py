"""
malta_fit_null.py

Null model for Malta: single-sigmoid's monotonic transmission shape,
with behavioral feedback disabled entirely (sensitivity/lambda_P FIXED
at 0.0, removed from the search, not just left free to converge there).
Mirrors exactly how Vetlyanka's null model was constructed (single's
shape, k=9, one fewer parameter than single's k=10, sensitivity
removed).

Purpose: directly test whether SOME suppression mechanism is essential
to explain Malta's mortality trajectory at all, the same decisive,
foundational check already confirmed for Vetlyanka (null SSE=66,131,
catastrophic failure, versus hundreds for every model that includes
feedback). This is a different, more basic question than the
double-vs-single comparison -- it establishes that behavioral
suppression is doing real, necessary work in Malta's own fit, not just
plausible.

Fits against DDI (plague-only cumulative deaths), not DR (all-cause).

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "fits", "null")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VETLYANKA_DIR = "/Users/bornalid/PycharmProjects/lower-volga-plague-regime-inference/vetlyanka "
sys.path.insert(0, VETLYANKA_DIR)

from plague_single_sigmoid_model import plague_model

# Single-sigmoid's shape, minus sensitivity (fixed at 0.0 -- no feedback at all)
PARAM_NAMES_NULL = ["b1", "b2", "x0", "c", "gamma1", "gamma2", "mu1", "mu2",
                     "T_perceive", "dispose_rate"]
BOUNDS_NULL = [
    (0.001, 8.0),    # b1
    (1.0, 50.0),     # b2 -- matches single.py/double.py exactly (shared-bounds comparison)
    (1.0, 20.0),     # x0 -- matches single.py/double.py exactly (shared-bounds comparison)
    (0.01, 5.0),     # c
    (0.35, 0.7),     # gamma1
    (1.0, 1.4),      # gamma2
    (1.0, 1.4),      # mu1
    (1.0, 10),       # mu2
    (0.1, 20.0),     # T_perceive -- kept in the search even though feedback is
                     # disabled: it still governs how P is computed and saved,
                     # even though P has no effect on transmission when
                     # sensitivity=0. Harmless either way -- included for
                     # structural consistency with the other Malta models.
    (0.1, 15.0),     # dispose_rate
]
K_NULL = len(PARAM_NAMES_NULL)
SENSITIVITY_FIXED = 0.0  # no behavioral feedback at all -- the whole point of this model

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


def reduced_to_full(reduced):
    d = dict(zip(PARAM_NAMES_NULL, reduced))
    d["sensitivity"] = SENSITIVITY_FIXED
    return np.array([d.get(name, X1_C1_PLACEHOLDER) for name in FULL_PARAM_NAMES])


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
    predicted = simulate_weekly(reduced)
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
    print(f"\n{'='*60}\nMalta null model (no feedback), N={num_samples}\n{'='*60}")
    sampler = qmc.LatinHypercube(d=K_NULL, seed=LHS_SEED)
    unit = sampler.random(n=num_samples)
    samples = np.array([[row[j] * (BOUNDS_NULL[j][1] - BOUNDS_NULL[j][0]) + BOUNDS_NULL[j][0]
                          for j in range(K_NULL)] for row in unit])

    scored = [(sse_unscaled(row), row) for row in samples]
    scored.sort(key=lambda p: p[0])
    top = scored[:TOP_K]

    best_sse, best_params = np.inf, None
    for lhs_sse, start in top:
        sx0, sf = scale_parameters(start)
        sb = [(lo/f, hi/f) for (lo, hi), f in zip(BOUNDS_NULL, sf)]
        result = minimize(lambda sp: sse_unscaled(sp * sf), sx0, method="L-BFGS-B", bounds=sb,
                           options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})
        opt = result.x * sf
        psse = sse_unscaled(opt)
        if psse < best_sse:
            best_sse, best_params = psse, opt

    pinned = check_pinned(best_params, PARAM_NAMES_NULL, BOUNDS_NULL)
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
    print(f"\nFinal null-model SSE (N={sizes[-1]}): {best_sse:.3f}")
    for name, val in zip(PARAM_NAMES_NULL, best_params):
        print(f"  {name:15s} = {val:.6f}")

    pd.DataFrame({"Parameter": PARAM_NAMES_NULL, "Bound_lo": [b[0] for b in BOUNDS_NULL],
                  "Bound_hi": [b[1] for b in BOUNDS_NULL], "Best_Fit": best_params}
                 ).to_csv(os.path.join(OUTPUT_DIR, "malta_null_polished_fit.csv"), index=False)

    # Save the consistency check itself, not just the final parameters -- this
    # was previously only ever printed to console, and got lost if the run
    # wasn't captured properly.
    with open(os.path.join(OUTPUT_DIR, "malta_null_run_summary.txt"), "w") as f:
        f.write(f"Malta null model run summary\n")
        f.write(f"N={sizes[0]}: SSE={results[sizes[0]][0]:.6f}\n")
        f.write(f"N={sizes[-1]}: SSE={results[sizes[-1]][0]:.6f}\n")
        f.write(f"SSE change between sample sizes: {pct:.4f}%\n")
        f.write(f"Consistent: {consistent}\n")
        f.write(f"Pinned parameters (N={sizes[-1]}): {pinned}\n")
    print(f"Saved {OUTPUT_DIR}/malta_null_run_summary.txt")

    print(f"\nCompare against every feedback-enabled Malta model (single=18,961,")
    print(f"single-constant=13,549, double=14,215.55, double-constant=8,953).")
    print(f"If null is dramatically worse than all of them, that directly confirms")
    print(f"behavioral suppression is essential to explain Malta's trajectory at all,")
    print(f"the same foundational result already established for Vetlyanka (null=66,131).")


if __name__ == "__main__":
    main()