"""
double_sigmoid_confidence_intervals.py   (step 07 -- run after 02_fits and 04 precompute_trajectory_library.py)

NEAR-OPTIMAL PARAMETER RANGES for the double-sigmoid (time-varying gamma, mu)
model. (The file keeps its old name so other scripts can find it.)

These are NOT confidence intervals. Method: score every valid candidate in the
precomputed trajectory library against the observed data, keep the best
PERCENTILE_BAND % by SSE, and report the 2.5th-97.5th percentile spread of each
parameter within that set. It answers "which parameter values occur among the
best-fitting sampled candidates", has no Hessian or quadratic extrapolation,
and can never leave a parameter's box bounds. It is not a sampling
distribution: the width depends on the chosen band and on the LHS density.

Only the MAIN library rows are used (the rows drawn within the real fit's
bounds). The widened-x1 rows added by expand_double_library_for_null_test.py
(x1 up to 40, outside the real bounds) are excluded.

The point estimate is the multi-start polished fit from 02_fits. It can sit
outside the band when polishing reached a better point than raw LHS sampling
did -- reported as-is, not an error.

Reads:  04_bootstrap/results/sigmoidal/double_trajectories.npy, candidate_params.npy
        02_fits/results/double/vetlyanka_polished_fit.csv
Writes: 07_uncertainty/results/double_sigmoid_confidence_intervals.csv
        (columns CI_lower_pctl / CI_upper_pctl kept for compatibility with the table scripts;
         they are the near-optimal range bounds)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # .../vetlyanka/07_uncertainty
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):  # walk up to vetlyanka/, which contains src/
    if os.path.dirname(ROOT) == ROOT:
        raise FileNotFoundError("Could not find the src/ folder above this script.")
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd

from vetlyanka_bounds import PARAM_NAMES

PERCENTILE_BAND = 1.0   # best 1% of library candidates by SSE
CI_LOWER_PCTL, CI_UPPER_PCTL = 2.5, 97.5

LIB_DIR = os.path.join(ROOT, "04_bootstrap", "results", "sigmoidal")
FIT_PATH = os.path.join(ROOT, "02_fits", "results", "double", "vetlyanka_polished_fit.csv")
OUTPUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

observed = np.array([3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19, 27, 34,
                     90, 259, 313, 345, 364, 376, 376, 376, 376], dtype=float)


def main():
    for p in (os.path.join(LIB_DIR, "double_trajectories.npy"), os.path.join(LIB_DIR, "candidate_params.npy")):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found -- run 04_bootstrap/precompute_trajectory_library.py first.")
    if not os.path.exists(FIT_PATH):
        raise FileNotFoundError(f"{FIT_PATH} not found -- run 02_fits/polish_vetlyanka_fit_double.py first.")

    params = np.load(os.path.join(LIB_DIR, "candidate_params.npy"))
    traj = np.load(os.path.join(LIB_DIR, "double_trajectories.npy"))[:params.shape[0]]   # main rows only
    ok = ~np.isnan(traj).any(axis=1)
    traj, params = traj[ok], params[ok]
    sse = np.sum((traj - observed) ** 2, axis=1)

    n_top = max(1, int(len(sse) * PERCENTILE_BAND / 100.0))
    top = np.argsort(sse)[:n_top]
    top_params, top_sse = params[top], sse[top]

    fit = pd.read_csv(FIT_PATH).set_index("Parameter")
    point = np.array([fit.loc[n, "Best_Fit"] for n in PARAM_NAMES], dtype=float)

    print(f"Library (main rows, within the real bounds): {len(sse)} valid candidates")
    print(f"Best {PERCENTILE_BAND:g}% band: {n_top} candidates, SSE {top_sse.min():.1f} to {top_sse.max():.1f}")
    print("These are NEAR-OPTIMAL RANGES, not confidence intervals.\n")

    rows = []
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = np.percentile(top_params[:, i], [CI_LOWER_PCTL, CI_UPPER_PCTL])
        rows.append({"Parameter": name, "Point_estimate": point[i],
                     "CI_lower_pctl": lo, "CI_upper_pctl": hi,
                     "point_outside_band": bool(point[i] < lo or point[i] > hi)})
    df = pd.DataFrame(rows)
    df.attrs["note"] = "near-optimal ranges"
    out = os.path.join(OUTPUT_DIR, "double_sigmoid_confidence_intervals.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))

    outside = df[df.point_outside_band]
    if len(outside):
        print(f"\nPoint estimate outside its near-optimal range for: {list(outside.Parameter)} "
              f"-- expected when polishing goes beyond what raw LHS sampling reached.")
    with open(os.path.join(OUTPUT_DIR, "double_sigmoid_confidence_intervals_README.txt"), "w") as fh:
        fh.write(f"Near-optimal ranges (NOT confidence intervals): 2.5-97.5 percentiles of each parameter "
                 f"among the best {PERCENTILE_BAND:g}% of {len(sse)} library candidates "
                 f"(n = {n_top}, SSE {top_sse.min():.1f}-{top_sse.max():.1f}; polished best fit SSE "
                 f"from 02_fits).\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()