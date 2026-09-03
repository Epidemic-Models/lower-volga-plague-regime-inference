"""
frozen_confidence_intervals.py

Top-1%-of-LHS-candidates percentile-band confidence intervals for the
frozen models, reusing the trajectory library and candidate parameters
already saved by build_frozen_library.py -- no new expensive simulation
needed, just re-ranking against the REAL observed data instead of
synthetic data.

Same method as Figs. 4/6 in the main text (percentile spread across the
best-fitting candidates), not the Hessian-inverse approach, which this
session already showed produces degenerate, unreliable intervals.

Requires:
    double_frozen_trajectories.npy   (from build_frozen_library.py)
    candidate_params_frozen_main.npy (from build_frozen_library.py, the
                                       MAIN batch only -- excludes the
                                       widened-x1 batch, which represents
                                       a deliberately different
                                       hypothesis space and shouldn't be
                                       mixed into the real fit's CI)
"""

import numpy as np

REDUCED_NAMES = ["b1", "b2", "x0", "x1", "c", "c1", "gamma_const", "mu_const",
                  "T_perceive", "sensitivity", "dispose_rate"]

observed_cumulative = np.array([
    3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19,
    27, 34, 90, 259, 313, 345, 364, 376,
    376, 376, 376
], dtype=float)

# Main batch only. candidate_params_frozen_main.npy has exactly as many
# rows as the main batch (the widened-x1 batch is never saved to this
# file), so slice double_frozen_trajectories.npy to match its length
# rather than hardcoding a sample count.
candidate_params = np.load("candidate_params_frozen_main.npy")
double_traj = np.load("double_frozen_trajectories.npy")[:candidate_params.shape[0]]

valid = ~np.isnan(double_traj).any(axis=1)
double_traj = double_traj[valid]
candidate_params = candidate_params[valid]
print(f"{candidate_params.shape[0]} valid double-frozen candidates available.")

sse = np.sum((double_traj - observed_cumulative) ** 2, axis=1)
n_top = max(int(0.01 * len(sse)), 1)
top_idx = np.argsort(sse)[:n_top]
top_params = candidate_params[top_idx]

print(f"\nTop {n_top} candidates (top 1%) -- percentile-band 95% CIs:")
print(f"{'Parameter':<15}{'Best-fit (real)':<18}{'2.5th pct':<14}{'97.5th pct':<14}")
best_fit_row = candidate_params[np.argmin(sse)]
for i, name in enumerate(REDUCED_NAMES):
    lo, hi = np.percentile(top_params[:, i], [2.5, 97.5])
    print(f"{name:<15}{best_fit_row[i]:<18.4f}{lo:<14.4f}{hi:<14.4f}")

print("\nNote: replace best_fit_row above with your ACTUAL polished (LHS+L-BFGS-B)")
print("best fit from fit_freeze_mu_gamma.py's own output for the true point estimate --")
print("this script's own argmin is only the best raw LHS candidate, not the polished one.")
