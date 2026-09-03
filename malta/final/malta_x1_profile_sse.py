"""
malta_x1_profile_sse.py

Profile-SSE analysis of the decline-onset parameter x1, for Malta's
sigmoidal double-regime model -- the direct analogue of the Vetlyanka
profile analysis (SI "Profile analysis of the decline-onset parameter").

Purpose: the full pairwise correlation screen (malta_full_correlation_screen.py)
found x1 strongly correlated (|r| up to 0.96) with several other parameters
among a near-optimal LHS pool. That result is consistent with either
(a) a genuinely flat/degenerate SSE surface along x1 -- the parameter
    really is poorly identified for Malta, unlike Vetlyanka -- or
(b) a lumpy, incompletely-explored LHS pool that never properly
    re-optimized the other 11 parameters conditional on each x1 value,
    producing spurious apparent correlation.
Only a profile-SSE sweep (fix x1, re-optimize everything else, repeat)
distinguishes these. This script performs exactly that sweep, following
the same procedure used for Vetlyanka: minimum-separation ordering
constraint enforced via profile-specific bounds, multiple starting points
per profile point (unrestricted best fit, a precomputed LHS library, and
the neighbouring profile point's own solution), lowest SSE retained.

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
DISPOSE_RATE_FIXED = 3.0

# Profile settings. Malta's plausible x1 range is wider than Vetlyanka's
# (correlation screen's near-optimal pool spanned ~16.2-23.2wk), so sweep
# generously beyond that on both sides to see the full shape of the SSE
# surface, not just the interior of the previously-observed spread.
PROFILE_MIN, PROFILE_MAX, PROFILE_STEP = 8.0, 26.0, 0.2
# Ordering constraint: only require x0 < x1 for this diagnostic profile,
# matching the (looser) constraint Vetlyanka's own profile analysis used
# -- NOT the stricter minimum-gap constraint used in the primary fits,
# since the point here is to let the other parameters compensate as
# freely as possible at each fixed x1.
MIN_GAP_FOR_PROFILE = 0.5

LIBRARY_SIZE = 20000  # one-off LHS library, reused as a candidate pool at every profile point
LHS_SEED = 42
N_STARTS_PER_POINT = 6  # how many of the library's best-for-this-x1 candidates to polish, per point

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
    predicted = simulate_weekly(reduced)
    if predicted is None:
        return 1e12
    sse = np.sum((observed - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def scale_parameters(params):
    sf = np.maximum(np.abs(params), 1e-3)
    return params / sf, sf


def polish(start, bounds_for_fit):
    """L-BFGS-B polish from one starting point, with x1 pinned via bounds."""
    sx0, sf = scale_parameters(start)
    sb = [(lo / f, hi / f) for (lo, hi), f in zip(bounds_for_fit, sf)]
    result = minimize(lambda sp: sse_unscaled(sp * sf), sx0, method="L-BFGS-B",
                       bounds=sb, options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})
    opt = result.x * sf
    return sse_unscaled(opt), opt


def build_library():
    """One-off LHS library used as a candidate pool at every profile point,
    matching the Vetlyanka procedure's reuse of a precomputed library."""
    print(f"Building one-off LHS library ({LIBRARY_SIZE} samples)...")
    sampler = qmc.LatinHypercube(d=K, seed=LHS_SEED)
    unit = sampler.random(n=LIBRARY_SIZE)
    samples = np.array([[row[j] * (BOUNDS[j][1] - BOUNDS[j][0]) + BOUNDS[j][0] for j in range(K)]
                         for row in unit])
    return samples


def unrestricted_best_fit(library):
    """Quick unrestricted polish, used as one of the multi-start seeds at every profile point."""
    scored = sorted(((sse_unscaled(row), row) for row in library[:5000]), key=lambda p: p[0])
    best_sse, best_x = None, None
    for lhs_sse, start in scored[:10]:
        s, x = polish(start, BOUNDS)
        if best_sse is None or s < best_sse:
            best_sse, best_x = s, x
    return best_sse, best_x


def main():
    library = build_library()
    unrestricted_sse, unrestricted_x = unrestricted_best_fit(library)
    print(f"Unrestricted best fit: SSE={unrestricted_sse:.2f}, x1={unrestricted_x[IDX_X1]:.3f}")

    x1_grid = np.round(np.arange(PROFILE_MIN, PROFILE_MAX + PROFILE_STEP / 2, PROFILE_STEP), 2)
    results = []
    prev_solution = unrestricted_x.copy()

    for x1_fixed in x1_grid:
        # bounds for this profile point: x1 pinned to a tiny window around x1_fixed,
        # x0 constrained below x1_fixed - MIN_GAP_FOR_PROFILE, everything else free
        bounds_for_fit = list(BOUNDS)
        bounds_for_fit[IDX_X1] = (x1_fixed - 1e-6, x1_fixed + 1e-6)
        x0_hi = min(BOUNDS[IDX_X0][1], x1_fixed - MIN_GAP_FOR_PROFILE)
        if x0_hi <= BOUNDS[IDX_X0][0]:
            print(f"x1={x1_fixed:.2f}: skipped, no feasible x0 range under min-gap constraint")
            continue
        bounds_for_fit[IDX_X0] = (BOUNDS[IDX_X0][0], x0_hi)

        # candidate starting points: unrestricted best fit, neighbouring
        # profile point's solution, and the N_STARTS_PER_POINT library
        # candidates whose x0 already satisfies the gap constraint (closest
        # in SSE to the running best, to keep this tractable)
        candidates = [unrestricted_x.copy(), prev_solution.copy()]
        feasible_lib = library[library[:, IDX_X0] < x0_hi]
        if len(feasible_lib) > 0:
            scored_lib = sorted(
                ((sse_unscaled(np.concatenate([row[:IDX_X1], [x1_fixed], row[IDX_X1 + 1:]])), row)
                 for row in feasible_lib[:2000]),
                key=lambda p: p[0]
            )
            candidates.extend(row for _, row in scored_lib[:N_STARTS_PER_POINT])

        best_sse_here, best_x_here = None, None
        for cand in candidates:
            cand = cand.copy()
            cand[IDX_X1] = x1_fixed
            cand[IDX_X0] = min(cand[IDX_X0], x0_hi - 1e-3)
            s, x = polish(cand, bounds_for_fit)
            if best_sse_here is None or s < best_sse_here:
                best_sse_here, best_x_here = s, x

        results.append({"x1": x1_fixed, "SSE": best_sse_here,
                         **{name: val for name, val in zip(PARAM_NAMES, best_x_here)}})
        prev_solution = best_x_here.copy()
        print(f"x1={x1_fixed:6.2f}  SSE={best_sse_here:10.2f}")

    df = pd.DataFrame(results)
    profile_min_row = df.loc[df["SSE"].idxmin()]
    df["delta_SSE"] = df["SSE"] - profile_min_row["SSE"]

    out_path = os.path.join(OUTPUT_DIR, "malta_x1_profile_sse.csv")
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print(f"Profile minimum: x1 = {profile_min_row['x1']:.2f}, SSE = {profile_min_row['SSE']:.2f}")
    print(f"Unrestricted best fit: x1 = {unrestricted_x[IDX_X1]:.3f}, SSE = {unrestricted_sse:.2f}")
    print("=" * 60)

    # Characterise the shape: how wide is the region within various SSE
    # tolerances of the profile minimum? A tight profile (Vetlyanka-like)
    # will have a narrow window even at generous tolerances; a flat/degenerate
    # profile (what the correlation screen's spread suggested for Malta)
    # will have a wide window even at tight tolerances.
    for tol_pct in (5, 10, 20):
        thresh = profile_min_row["SSE"] * (1 + tol_pct / 100)
        within = df[df["SSE"] <= thresh]
        if len(within) > 0:
            print(f"Within {tol_pct}% of minimum SSE: x1 in [{within['x1'].min():.2f}, "
                  f"{within['x1'].max():.2f}]  (width = {within['x1'].max() - within['x1'].min():.2f} weeks, "
                  f"{len(within)} grid points)")
        else:
            print(f"Within {tol_pct}% of minimum SSE: no other grid points")

    print(f"\nSaved {out_path}")
    print("\nPlot delta_SSE vs x1 to compare directly against Vetlyanka's Fig. S3 shape.")


if __name__ == "__main__":
    main()