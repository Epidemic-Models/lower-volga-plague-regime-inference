"""
double_sigmoid_confidence_intervals.py

Percentile-band confidence intervals for the sigmoidal DOUBLE-sigmoid
model's parameters, using the same method already built and validated
for the frozen pair (frozen_confidence_intervals.py) -- NOT the
Hessian-based approximation in polish_vetlyanka_fit_double.py's own
output CSV, which produces meaningless, unbounded intervals whenever a
parameter sits along a flat/degenerate direction of the fit surface
(confirmed directly: T_perceive's Hessian-based CI reported an upper
bound of 432, far outside its own (0.5, 20) box constraint, and every
parameter's CI_lower was identically 0.0 -- the tell that the whole
approximation had broken down, not that thirteen parameters happen to
share a real lower bound of exactly zero).

METHOD: reuse the already-built trajectory library
(double_trajectories.npy / candidate_params.npy, from
precompute_trajectory_library.py -- these already respect the x0/x1
ordering constraint). Score every valid candidate's SSE against the
real observed data, take the top 1% by SSE, and report the percentile
spread of each parameter WITHIN that top 1% as the interval. This has
no Hessian, no local quadratic extrapolation, and cannot report a value
outside a parameter's own box bound, since every candidate in the
library was drawn from within that bound to begin with.

The REAL point estimate reported is the actual multi-start polished fit
from vetlyanka_polished_fit.csv (polish_vetlyanka_fit_double.py's true
output), not the crude argmin-of-raw-library value -- the library is
LHS-only, unpolished, so its own best candidate is not as good as the
true L-BFGS-B-polished optimum. It is normal and expected for the real
point estimate to sit slightly outside the top-1% percentile band in
some cases (the polish step can push past what raw LHS sampling alone
reached) -- report this as-is rather than treating it as an error, the
same way it was already handled for the frozen pair.

Requires:
    double_trajectories.npy, candidate_params.npy (from
        precompute_trajectory_library.py)
    vetlyanka_polished_fit.csv (the real point estimate)
"""

import numpy as np
import pandas as pd

from vetlyanka_bounds import PARAM_NAMES

PERCENTILE_BAND = 1.0  # top 1% of candidates by SSE
CI_LOWER_PCTL = 2.5
CI_UPPER_PCTL = 97.5

observed = np.array([3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19, 27, 34,
                      90, 259, 313, 345, 364, 376, 376, 376, 376], dtype=float)


def main():
    print("Loading trajectory library...")
    trajectories = np.load("double_trajectories.npy")
    candidate_params = np.load("candidate_params.npy")

    valid_mask = ~np.isnan(trajectories).any(axis=1)
    trajectories = trajectories[valid_mask]
    candidate_params = candidate_params[valid_mask]
    print(f"Using {trajectories.shape[0]} valid candidates "
          f"(x0/x1-degenerate and failed-solve candidates already excluded).")

    sse = np.sum((trajectories - observed) ** 2, axis=1)

    n_top = max(1, int(len(sse) * PERCENTILE_BAND / 100.0))
    top_idx = np.argsort(sse)[:n_top]
    top_params = candidate_params[top_idx]
    top_sse = sse[top_idx]
    print(f"Top {PERCENTILE_BAND}% band: {n_top} candidates, "
          f"SSE range [{top_sse.min():.3f}, {top_sse.max():.3f}]")

    real_fit_df = pd.read_csv("vetlyanka_polished_fit.csv").set_index("Parameter")
    real_point_estimate = np.array([real_fit_df.loc[name, "Best_Fit"] for name in PARAM_NAMES])

    results = []
    for i, name in enumerate(PARAM_NAMES):
        band_values = top_params[:, i]
        ci_lower = np.percentile(band_values, CI_LOWER_PCTL)
        ci_upper = np.percentile(band_values, CI_UPPER_PCTL)
        point_est = real_point_estimate[i]
        outside_band = point_est < ci_lower or point_est > ci_upper
        results.append({
            "Parameter": name,
            "Point_estimate": point_est,
            "CI_lower_pctl": ci_lower,
            "CI_upper_pctl": ci_upper,
            "point_outside_band": outside_band,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("double_sigmoid_confidence_intervals.csv", index=False)

    print("\n", results_df.to_string(index=False))

    outside = results_df[results_df["point_outside_band"]]
    if len(outside) > 0:
        print(f"\nNOTE: point estimate sits outside its own top-{PERCENTILE_BAND}% band for: "
              f"{list(outside['Parameter'])}")
        print("This is expected when the gradient-polished optimizer finds a better point than")
        print("raw LHS sampling reached on its own -- report as-is, not an error.")

    print("\nSaved: double_sigmoid_confidence_intervals.csv")


if __name__ == "__main__":
    main()