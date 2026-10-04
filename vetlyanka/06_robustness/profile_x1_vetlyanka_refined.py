"""
profile_x1_vetlyanka_refined.py   (step 06 -- run after 02_fits and 04 precompute/expand library)

Profile-SSE analysis for the decline-onset parameter x1 of the standalone
Vetlyanka double-sigmoid model.

For each fixed x1 on a grid (X1_STEP apart over x1's bounds):
  * x1 is held fixed and the other 12 parameters are re-optimized with
    L-BFGS-B (scaled parameters, as in the 02_fits scripts);
  * starts: the neighbouring grid point's solution (warm start / continuation),
    the unrestricted best fit with x1 replaced, and the best library candidates
    whose x1 is closest to the grid value;
  * the SAME constraint as the real fit is enforced: x1 - x0 >= X0_X1_MIN_GAP
    (1 week). (The previous version only required x0 < x1.)
The minimum SSE per grid point is the profile.

This is a PROFILE-SSE analysis, not a formal confidence interval: successive
cumulative observations are dependent, so a chi-squared threshold is not
justified without a validated observation model.

Reads:
  02_fits/results/double/vetlyanka_polished_fit.csv
  04_bootstrap/results/sigmoidal/ candidate_params.npy (+ candidate_params_widened_x1.npy),
                                  double_trajectories.npy
Writes 06_robustness/results/x1_profile/:
  x1_profile_sse_refined.csv, x1_profile_optimization_log_refined.csv, x1_profile_sse_refined.png
  (partial CSVs are written after every grid point, so a stopped run keeps its progress)
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from vetlyanka_bounds import PARAM_NAMES, BOUNDS
from vetlyanka_simulation import simulate_vetlyanka, LEGACY_ALIGNMENT

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
X1_STEP = 0.1
N_LIBRARY_STARTS = 4        # + warm start + unrestricted fit = up to 6 polishes per grid point
CANDIDATE_POOL_SIZE = 2000  # best library candidates considered for starts
MAXITER = 5000
X0_X1_MIN_GAP = 1.0         # must match polish_vetlyanka_fit_double.py

OBSERVED = np.array([3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19, 27, 34,
                     90, 259, 313, 345, 364, 376, 376, 376, 376], dtype=float)

FIT_PATH = os.path.join(ROOT, "02_fits", "results", "double", "vetlyanka_polished_fit.csv")
LIB_DIR = os.path.join(ROOT, "04_bootstrap", "results", "sigmoidal")
OUTPUT_DIR = os.path.join(HERE, "results", "x1_profile")
os.makedirs(OUTPUT_DIR, exist_ok=True)

X1_IDX = PARAM_NAMES.index("x1")
X0_IDX = PARAM_NAMES.index("x0")
X1_GRID = np.round(np.arange(BOUNDS[X1_IDX][0], BOUNDS[X1_IDX][1] + X1_STEP / 2, X1_STEP), 10)
FREE_IDX = [i for i in range(len(PARAM_NAMES)) if i != X1_IDX]
PENALTY = 1e12


def sse(full):
    if full[X0_IDX] > full[X1_IDX] - X0_X1_MIN_GAP:
        return PENALTY
    pred = simulate_vetlyanka(full)
    if pred is None:
        return PENALTY
    v = float(np.sum((pred - OBSERVED) ** 2))
    return v if np.isfinite(v) else PENALTY


def free_bounds(x1):
    b = [BOUNDS[i] for i in FREE_IDX]
    j = FREE_IDX.index(X0_IDX)
    lo, hi = b[j]
    b[j] = (lo, min(hi, x1 - X0_X1_MIN_GAP))   # enforce x1 - x0 >= gap through the bounds
    if b[j][1] <= lo:
        raise ValueError(f"No feasible x0 for x1 = {x1}")
    return b


def full_from_free(free, x1):
    full = np.empty(len(PARAM_NAMES)); full[FREE_IDX] = free; full[X1_IDX] = x1
    return full


def polish(start_free, x1, bounds):
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    start = np.clip(start_free, lo, hi)
    sf = np.maximum(np.abs(start), 1e-3)
    res = minimize(lambda p: sse(full_from_free(p * sf, x1)), start / sf, method="L-BFGS-B",
                   bounds=list(zip(lo / sf, hi / sf)),
                   options={"maxiter": MAXITER, "ftol": 1e-12, "gtol": 1e-10, "maxls": 100})
    p = res.x * sf
    return sse(full_from_free(p, x1)), full_from_free(p, x1), res


def load_inputs():
    if not os.path.exists(FIT_PATH):
        raise FileNotFoundError(f"{FIT_PATH} not found -- run 02_fits/polish_vetlyanka_fit_double.py first.")
    fit = pd.read_csv(FIT_PATH).set_index("Parameter")
    best = np.array([fit.loc[n, "Best_Fit"] for n in PARAM_NAMES], dtype=float)

    traj = np.load(os.path.join(LIB_DIR, "double_trajectories.npy"))
    params = np.load(os.path.join(LIB_DIR, "candidate_params.npy"))
    wide = os.path.join(LIB_DIR, "candidate_params_widened_x1.npy")
    if traj.shape[0] > params.shape[0] and os.path.exists(wide):
        params = np.vstack([params, np.load(wide)])
    if traj.shape[0] != params.shape[0]:
        raise ValueError("Library rows and parameter rows do not line up -- rebuild the library (04_bootstrap).")
    ok = np.isfinite(traj).all(axis=1)
    params, traj = params[ok], traj[ok]
    lib_sse = np.sum((traj - OBSERVED) ** 2, axis=1)
    pool = params[np.argsort(lib_sse)[:CANDIDATE_POOL_SIZE]]
    return best, pool


def save(results, log, final=False):
    tag = "refined" if final else "partial"
    rows = [{"x1": x1, "profile_sse": r["sse"], **dict(zip(PARAM_NAMES, r["params"]))}
            for x1, r in sorted(results.items())]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, f"x1_profile_sse_{tag}.csv"), index=False)
    pd.DataFrame(log).to_csv(os.path.join(OUTPUT_DIR, f"x1_profile_optimization_log_{tag}.csv"), index=False)
    return df


def profile_point(x1, warm, best, pool, log):
    bounds = free_bounds(x1)
    starts = [("warm", np.delete(warm, X1_IDX)), ("unrestricted_fit", np.delete(best, X1_IDX))]
    near = pool[np.argsort(np.abs(pool[:, X1_IDX] - x1))[:N_LIBRARY_STARTS]]
    starts += [(f"library_{i + 1}", np.delete(r, X1_IDX)) for i, r in enumerate(near)]
    best_s, best_p = np.inf, None
    for label, s in starts:
        v, p, res = polish(s, x1, bounds)
        log.append({"x1": x1, "start": label, "SSE": v, "iterations": res.nit})
        if v < best_s:
            best_s, best_p = v, p
    return best_s, best_p


def main():
    best, pool = load_inputs()
    best_sse = sse(best)
    print(f"Observation times: {'t = 0..21 (LEGACY)' if LEGACY_ALIGNMENT else 't = 1..22 (end of each week)'}")
    print(f"Unrestricted fit: x1 = {best[X1_IDX]:.4f}, SSE = {best_sse:.6f}")
    print(f"Grid: x1 = {X1_GRID[0]:g} .. {X1_GRID[-1]:g} step {X1_STEP} ({len(X1_GRID)} points), "
          f"x1 - x0 >= {X0_X1_MIN_GAP}, up to {N_LIBRARY_STARTS + 2} polishes per point\n")

    centre = int(np.argmin(np.abs(X1_GRID - best[X1_IDX])))
    results, log, t0 = {}, [], time.time()
    order = [centre] + list(range(centre - 1, -1, -1)) + list(range(centre + 1, len(X1_GRID)))
    warm = best.copy()
    for n, idx in enumerate(order):
        if idx == centre - 1 or idx == centre + 1:
            warm = results[X1_GRID[centre]]["params"]         # restart continuation from the centre
        x1 = X1_GRID[idx]
        s, p = profile_point(x1, warm, best, pool, log)
        results[x1] = {"sse": s, "params": p}
        warm = p
        save(results, log)
        print(f"  [{n + 1:2d}/{len(X1_GRID)}] x1 = {x1:5.2f}  profile SSE = {s:10.4f}   "
              f"[{(time.time() - t0) / 60:.0f} min]", flush=True)

    df = save(results, log, final=True)
    df["delta_sse"] = df.profile_sse - df.profile_sse.min()
    df.to_csv(os.path.join(OUTPUT_DIR, "x1_profile_sse_refined.csv"), index=False)
    b = df.loc[df.profile_sse.idxmin()]

    plt.figure(figsize=(7, 5))
    plt.plot(df.x1, df.delta_sse, marker="o", markersize=3)
    plt.axvline(b.x1, linestyle="--", label=rf"Profile minimum $x_1={b.x1:.2f}$")
    plt.axvline(best[X1_IDX], linestyle=":", label=rf"Unrestricted fit $x_1={best[X1_IDX]:.2f}$")
    plt.xlabel(r"Fixed decline-onset parameter $x_1$ (weeks)")
    plt.ylabel(r"$\Delta$SSE relative to profile minimum")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "x1_profile_sse_refined.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nUnrestricted fit: x1 = {best[X1_IDX]:.4f}, SSE = {best_sse:.4f}")
    print(f"Profile minimum:  x1 = {b.x1:.2f}, SSE = {b.profile_sse:.4f}")
    if b.profile_sse < best_sse - 1e-6:
        print("NOTE: the profile found a LOWER SSE than the unrestricted fit -- the unrestricted search "
              "missed this; report the profile minimum.")
    print("\nBest 15 profile points:")
    print(df.sort_values("profile_sse")[["x1", "profile_sse", "delta_sse"]].head(15).to_string(index=False))
    for frac in (0.05, 0.10, 0.20):
        inside = df[df.profile_sse <= b.profile_sse * (1 + frac)]
        print(f"x1 within {int(frac * 100)}% of the minimum SSE: {inside.x1.min():.2f} to {inside.x1.max():.2f}")
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()