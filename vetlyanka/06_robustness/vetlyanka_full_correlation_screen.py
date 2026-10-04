"""
vetlyanka_full_correlation_screen.py   (step 06 -- run after 02_fits)

Correlation screen of near-optimal double-sigmoid fits for Vetlyanka
(Reviewer #3: how were correlated parameter estimates handled in practice?).

Procedure: LHS scan, polish the TOP_K_TO_POLISH best candidates (same settings
as the real fit, x1 - x0 >= 1), keep every polished fit within
QUALITY_THRESHOLD of the best SSE found, and report pairwise correlations
across that near-optimal pool -- in particular, how x1 (the shift time the
paper's claim depends on) co-varies with the other parameters.

Caveat: the pool is at most TOP_K_TO_POLISH fits, so correlations are
descriptive of the near-optimal set, not a formal posterior.

Bounds are imported from src/vetlyanka_bounds.py (no hand-typed copy).
OBSERVATION TIMES: weeks 1..22 at t = 1..22 (same as the fits).

Reads:  02_fits/results/double/vetlyanka_polished_fit.csv (reference SSE)
Writes: 06_robustness/results/correlation_screen/
          vetlyanka_near_optimal_pool.csv, vetlyanka_parameter_correlation_matrix.csv
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # .../vetlyanka/06_robustness
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):  # walk up to vetlyanka/, which contains src/
    if os.path.dirname(ROOT) == ROOT:
        raise FileNotFoundError("Could not find the src/ folder above this script.")
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import qmc

from plague_double_sigmoid_model import plague_model
from vetlyanka_bounds import PARAM_NAMES, BOUNDS

LEGACY_ALIGNMENT = False
QUALITY_THRESHOLD = 1.20     # keep polished fits within 20% of the best SSE found
NUM_LHS_SAMPLES = 100000
TOP_K_TO_POLISH = 150
LHS_SEED = 42
X0_X1_MIN_GAP = 1.0          # must match polish_vetlyanka_fit_double.py

FIT_PATH = os.path.join(ROOT, "02_fits", "results", "double", "vetlyanka_polished_fit.csv")
OUTPUT_DIR = os.path.join(HERE, "results", "correlation_screen")
os.makedirs(OUTPUT_DIR, exist_ok=True)

observed = np.array([3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19, 27, 34, 90, 259,
                     313, 345, 364, 376, 376, 376, 376], dtype=float)
n_weeks = 22
t_start, t_end, dt = 0.0, 22.0, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
OBS = slice(0, n_weeks) if LEGACY_ALIGNMENT else slice(1, n_weeks + 1)
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]
IDX_X0, IDX_X1 = PARAM_NAMES.index("x0"), PARAM_NAMES.index("x1")
LO = np.array([b[0] for b in BOUNDS], float); HI = np.array([b[1] for b in BOUNDS], float)


def sse(params):
    if params[IDX_X0] >= params[IDX_X1] - X0_X1_MIN_GAP:
        return 1e12
    try:
        sol = solve_ivp(plague_model, [t_start, t_end], initial_conditions,
                        args=(params,), t_eval=t_points, method="RK45")
    except (ValueError, FloatingPointError, OverflowError):
        return 1e12
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return 1e12
    pred = sol.y[5][::steps_per_week][OBS]
    if pred.shape[0] != n_weeks:
        return 1e12
    v = float(np.sum((observed - pred) ** 2))
    return v if np.isfinite(v) else 1e12


def main():
    ref = pd.read_csv(FIT_PATH).set_index("Parameter") if os.path.exists(FIT_PATH) else None
    ref_sse = sse(np.array([ref.loc[n, "Best_Fit"] for n in PARAM_NAMES])) if ref is not None else np.nan
    print(f"Observation times: {'t = 0..21 (LEGACY)' if LEGACY_ALIGNMENT else 't = 1..22 (end of each week)'}")
    print(f"Reference (02_fits double): SSE = {ref_sse:.3f}")
    print(f"Drawing {NUM_LHS_SAMPLES} LHS samples, polishing the top {TOP_K_TO_POLISH}...", flush=True)

    samples = LO + qmc.LatinHypercube(d=len(BOUNDS), seed=LHS_SEED).random(n=NUM_LHS_SAMPLES) * (HI - LO)
    scores = np.array([sse(r) for r in samples])
    polished, t0 = [], time.time()
    for n, i in enumerate(np.argsort(scores)[:TOP_K_TO_POLISH]):
        start = samples[i]
        sf = np.maximum(np.abs(start), 1e-3)
        res = minimize(lambda p: sse(p * sf), start / sf, method="L-BFGS-B",
                       bounds=list(zip(LO / sf, HI / sf)),
                       options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 100})
        p = res.x * sf
        polished.append((sse(p), p))
        print(f"  polished {n + 1}/{TOP_K_TO_POLISH}: SSE = {polished[-1][0]:.3f}  "
              f"[{(time.time() - t0) / 60:.0f} min]", flush=True)

    best = min(s for s, _ in polished)
    pool = [(s, p) for s, p in polished if s <= best * QUALITY_THRESHOLD]
    print(f"\nBest SSE in this screen: {best:.3f}  (02_fits reference: {ref_sse:.3f})")
    print(f"Near-optimal pool (within {int((QUALITY_THRESHOLD - 1) * 100)}% of best): {len(pool)} fits")

    df = pd.DataFrame([p for _, p in pool], columns=PARAM_NAMES)
    df["SSE"] = [s for s, _ in pool]
    df.to_csv(os.path.join(OUTPUT_DIR, "vetlyanka_near_optimal_pool.csv"), index=False)
    if len(pool) < 3:
        print("Too few near-optimal fits for correlations -- increase TOP_K_TO_POLISH or QUALITY_THRESHOLD.")
        return
    corr = df[PARAM_NAMES].corr()
    corr.to_csv(os.path.join(OUTPUT_DIR, "vetlyanka_parameter_correlation_matrix.csv"))

    print("\nx1 across the near-optimal pool: "
          f"mean {df.x1.mean():.3f}, sd {df.x1.std():.3f}, range {df.x1.min():.3f} to {df.x1.max():.3f}")
    print("\nStrong correlations (|r| > 0.6):")
    strong = [(a, b, corr.loc[a, b]) for i, a in enumerate(PARAM_NAMES) for b in PARAM_NAMES[i + 1:]
              if np.isfinite(corr.loc[a, b]) and abs(corr.loc[a, b]) > 0.6]
    for a, b, r in strong:
        print(f"  {a} <-> {b}: r = {r:.2f}")
    if not strong:
        print("  None found.")
    print("\nx1 correlations with the other parameters:")
    print(corr.loc["x1"].drop("x1").sort_values(key=abs, ascending=False).round(3).to_string())
    print(f"\nSaved outputs to {OUTPUT_DIR}  (NaN correlations = parameter constant across the pool, e.g. at a bound)")


if __name__ == "__main__":
    main()