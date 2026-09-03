"""
malta_full_correlation_screen.py

Systematically screens ALL pairwise parameter correlations among
near-optimal candidate solutions for Malta's sigmoidal double model,
rather than testing individual pairs by hand. Any real trade-off
(like the already-found b2/dispose_rate and b1-b2/sensitivity ones)
will show up as a strong correlation among solutions that all achieve
similarly low SSE.

Reuses the confirmed dispose_rate=3.0 setup. Draws a large LHS pool,
keeps every candidate within a generous SSE-quality threshold of the
best found, and reports the full pairwise correlation matrix among
that near-optimal set.

Required:
    plague_double_sigmoid_model.py in the vetlyanka/ folder.
"""

import os
import sys
import signal
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import qmc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VETLYANKA_DIR = "/Users/bornalid/PycharmProjects/lower-volga-plague-regime-inference/vetlyanka "
sys.path.insert(0, VETLYANKA_DIR)
from plague_double_sigmoid_model import plague_model

PARAM_NAMES = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2", "mu1", "mu2",
               "T_perceive", "sensitivity"]
BOUNDS = [
    (0.001, 8.0), (1.0, 100.0), (1.0, 20.0), (5.0, 27.0), (0.01, 5.0), (0.01, 5.0),
    (0.35, 0.7), (1.0, 1.4), (1.0, 1.4), (1.0, 10), (0.1, 20.0), (0.0001, 0.5),
]
K = len(BOUNDS)
IDX_X0, IDX_X1 = 2, 3
X0_X1_MIN_GAP = 1.0
DISPOSE_RATE_FIXED = 3.0

QUALITY_THRESHOLD = 1.20  # keep every candidate within 20% of the best SSE found
NUM_LHS_SAMPLES = 60000
TOP_K_TO_POLISH = 30  # polish more than usual -- want a genuinely broad near-optimal pool
LHS_SEED = 42

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


def reduced_to_full(reduced):
    return np.concatenate([reduced, [DISPOSE_RATE_FIXED]])


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
    predicted = sol.y[4][::steps_per_week][:n_weeks]
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
    print(f"Best SSE found: {best_sse:.2f}")
    print(f"Near-optimal pool (within {int((QUALITY_THRESHOLD-1)*100)}%): {len(near_optimal)} candidates")

    if len(near_optimal) < 5:
        print("WARNING: too few near-optimal candidates to compute meaningful correlations.")
        print("Consider increasing QUALITY_THRESHOLD or TOP_K_TO_POLISH.")

    df = pd.DataFrame([p[1] for p in near_optimal], columns=PARAM_NAMES)
    df["SSE"] = [p[0] for p in near_optimal]
    df.to_csv(os.path.join(OUTPUT_DIR, "malta_near_optimal_pool.csv"), index=False)

    corr = df[PARAM_NAMES].corr()
    print("\nFull pairwise correlation matrix among near-optimal solutions:")
    print(corr.round(2))
    corr.to_csv(os.path.join(OUTPUT_DIR, "malta_parameter_correlation_matrix.csv"))

    print("\nStrong correlations (|r| > 0.6), candidate additional trade-offs:")
    found_any = False
    for i, p1 in enumerate(PARAM_NAMES):
        for p2 in PARAM_NAMES[i+1:]:
            r = corr.loc[p1, p2]
            if abs(r) > 0.6:
                print(f"  {p1} <-> {p2}: r = {r:.2f}")
                found_any = True
    if not found_any:
        print("  None found beyond the already-known b1/b2-sensitivity relationship.")

    print("\n" + "="*60)
    print("THE QUESTION THAT ACTUALLY MATTERS: is x1 (decline timing) entangled")
    print("with anything? This is the parameter the paper's claim depends on --")
    print("everything else in this matrix is diagnostic curiosity by comparison.")
    print("="*60)
    x1_correlations = corr.loc["x1"].drop("x1").sort_values(key=abs, ascending=False)
    print(x1_correlations.round(3))
    max_x1_corr = x1_correlations.abs().max()
    if max_x1_corr > 0.5:
        print(f"\nFLAG: x1 shows a real correlation (|r|={max_x1_corr:.2f}) -- this needs")
        print("direct attention before treating the decline-timing claim as robust.")
    else:
        print(f"\nx1 is NOT meaningfully correlated with anything else (max |r|={max_x1_corr:.2f}) --")
        print("a real, positive confirmation that the parameter Malta's finding depends")
        print("on is clean, independent of the b1/b2-sensitivity trade-off elsewhere.")

    print(f"\nSaved {OUTPUT_DIR}/malta_near_optimal_pool.csv")
    print(f"Saved {OUTPUT_DIR}/malta_parameter_correlation_matrix.csv")


if __name__ == "__main__":
    main()