"""
frozen_confidence_intervals_single.py

Same method as frozen_confidence_intervals.py, applied to single-frozen.
Reuses the SAME 200,000 candidates (candidate_params_frozen_main.npy) --
single_frozen_trajectories.npy was built by evaluating those exact same
candidates under plague_model_single instead of plague_model_double, so
no new sampling is needed. x1 and c1 are inert for the single-sigmoid
model and are omitted from the reported CIs.

Requires:
    single_frozen_trajectories.npy   (from build_frozen_library.py)
    candidate_params_frozen_main.npy (from build_frozen_library.py)
"""

import numpy as np

# Only the 9 parameters that actually matter for the single-sigmoid model.
# Indices refer to columns in candidate_params_frozen_main.npy, which is
# ordered: b1,b2,x0,x1,c,c1,gamma_const,mu_const,T_perceive,sensitivity,dispose_rate
SINGLE_RELEVANT_NAMES = ["b1", "b2", "x0", "c", "gamma_const", "mu_const",
                          "T_perceive", "sensitivity", "dispose_rate"]
SINGLE_RELEVANT_COLS = [0, 1, 2, 4, 6, 7, 8, 9, 10]  # skips x1 (3) and c1 (5)

observed_cumulative = np.array([
    3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19,
    27, 34, 90, 259, 313, 345, 364, 376,
    376, 376, 376
], dtype=float)

candidate_params = np.load("candidate_params_frozen_main.npy")
single_traj = np.load("single_frozen_trajectories.npy")[:candidate_params.shape[0]]

valid = ~np.isnan(single_traj).any(axis=1)
single_traj = single_traj[valid]
candidate_params = candidate_params[valid]
print(f"{candidate_params.shape[0]} valid single-frozen candidates available.")

sse = np.sum((single_traj - observed_cumulative) ** 2, axis=1)
n_top = max(int(0.01 * len(sse)), 1)
top_idx = np.argsort(sse)[:n_top]
top_params = candidate_params[top_idx]

print(f"\nTop {n_top} candidates (top 1%) -- percentile-band 95% CIs:")
print(f"{'Parameter':<15}{'Best-fit (real)':<18}{'2.5th pct':<14}{'97.5th pct':<14}")
best_fit_row = candidate_params[np.argmin(sse)]
for name, col in zip(SINGLE_RELEVANT_NAMES, SINGLE_RELEVANT_COLS):
    lo, hi = np.percentile(top_params[:, col], [2.5, 97.5])
    print(f"{name:<15}{best_fit_row[col]:<18.4f}{lo:<14.4f}{hi:<14.4f}")

print("\nNote: replace best_fit_row above with your ACTUAL polished best fit from")
print("polish_vetlyanka_fit_single_freeze_mu_gamma_FIXED.py's own output for the")
print("true point estimate -- this script's own argmin is only the best raw LHS")
print("candidate, not the polished one. Also worth checking: does T_perceive's")
print("97.5th percentile bump right up against 8.0 (its bound), consistent with")
print("the point estimate pinning we already flagged?")
