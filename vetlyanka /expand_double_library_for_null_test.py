"""
expand_double_library_for_null_test.py

Your double-sigmoid LHS library, built under the current real-fit bounds,
can only represent curves whose decline (x1) falls somewhere inside those
bounds -- it has no way to represent a near-monotonic curve with no
decline at all inside the 22-week window. For fitting the REAL data,
that's fine -- you have reason to expect the shift around then. But the
parametric bootstrap null test needs the double-sigmoid candidate pool
to have a genuine chance of representing "no second regime" when that's
actually true, or the test is rigged to fail regardless of what's really
going on. This script generates that supplementary batch, with x1 pushed
out to 40 (well past the 22-week observation window).

X0/X1 ORDERING CONSTRAINT: candidates where the rise-midpoint (x0) isn't
meaningfully before the decline-onset (x1) are excluded here too -- same
constraint as polish_vetlyanka_fit_double.py, precompute_trajectory_library.py,
and build_frozen_library.py. This batch needs it MORE than the main
library does, not less: x0 currently reaches up to 20, and this widened
x1 still starts as low as 14, so a meaningful slice of this deliberately
wide draw would otherwise land right back in the degenerate overlap the
other scripts now exclude.

gamma1/gamma2/mu1/mu2 are left untouched -- these are historically
grounded (documented death/recovery timescales), so their current real
bounds are correct as-is and don't need a "widened for the null test"
version.

IMPORTANT -- this script alone is not enough. It only builds the
SUPPLEMENTARY widened-x1 batch. The base double_trajectories.npy /
single_trajectories.npy libraries (from precompute_trajectory_library.py)
must already reflect the current bounds and fits. Before running this
script:
  1. Regenerate parameter_samples_<NUM_SAMPLES>.csv via parameter_sampling.py
     under the CURRENT vetlyanka_bounds.py bounds, if not already current.
  2. Re-run precompute_trajectory_library.py against that file (this now
     applies the same x0/x1 exclusion to its own double-sigmoid rows).
  3. THEN run this script to append the widened-x1 supplementary batch.
  4. THEN re-run parametric_bootstrap_test.py, using your latest
     polished single-sigmoid best fit (vetlyanka_polished_fit_single.csv)
     as ground truth, and a freshly-refreshed `observed` dict from
     AIC_AICc_BIC.py.

Run this once, after step 2 above and before step 4.
"""

import numpy as np
from scipy.integrate import solve_ivp
from pydoe import lhs
import gc

from plague_double_sigmoid_model import plague_model as plague_model_double
from vetlyanka_bounds import PARAMETERS, PARAM_NAMES

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "bootstrap", "sigmoidal")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Widened bounds -- built FROM the shared PARAMETERS (so b1/c/c1/
# T_perceive/dispose_rate/etc. can never silently drift out of sync
# with the real fit again), with only x1 overridden to push well
# beyond its normal bound so the null-test pool can represent "no
# decline at all" within the 22-week observation window.
# ------------------------------------------------------------
PARAMETERS_WIDENED = [
    (name, lo, 40 if name == 'x1' else hi)
    for name, lo, hi in PARAMETERS
]

NUM_NEW_SAMPLES = 50000  # 1:4 ratio to the main 200,000-sample library,
                          # matching the same proportion used for
                          # build_frozen_library.py's widened batch

X0_X1_MIN_GAP = 1.0  # must match polish_vetlyanka_fit_double.py
IDX_X0 = PARAM_NAMES.index("x0")
IDX_X1 = PARAM_NAMES.index("x1")

t_start, t_end, dt = 0, 22, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
n_weeks = 22
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]

# ------------------------------------------------------------
# Generate the widened-bound LHS samples
# ------------------------------------------------------------
num_params = len(PARAMETERS_WIDENED)
lhs_unit = lhs(num_params, samples=NUM_NEW_SAMPLES)
new_samples = np.array([
    [row[j] * (PARAMETERS_WIDENED[j][2] - PARAMETERS_WIDENED[j][1]) + PARAMETERS_WIDENED[j][1]
     for j in range(num_params)]
    for row in lhs_unit
])

print(f"Simulating {NUM_NEW_SAMPLES} widened-x1 double-sigmoid candidates "
      f"(excluding x1 - x0 < {X0_X1_MIN_GAP} weeks)...")


def simulate_weekly(model_fn, params):
    sol = solve_ivp(
        model_fn, [t_start, t_end], initial_conditions,
        args=(params,), t_eval=t_points, method="RK45",
    )
    if not sol.success:
        return None
    DR = sol.y[5]
    weekly = DR[::steps_per_week][:n_weeks]
    if len(weekly) < n_weeks:
        return None
    return weekly


new_trajectories = np.full((NUM_NEW_SAMPLES, n_weeks), np.nan)
n_rejected = 0
for i, params in enumerate(new_samples):
    if i % 1000 == 0 and i > 0:
        print(f"  sample {i}/{NUM_NEW_SAMPLES}  (rejected so far: {n_rejected})")
        gc.collect()

    if params[IDX_X0] >= params[IDX_X1] - X0_X1_MIN_GAP:
        n_rejected += 1
        continue

    w = simulate_weekly(plague_model_double, params)
    if w is not None:
        new_trajectories[i, :] = w

n_failed = int(np.isnan(new_trajectories).any(axis=1).sum())
print(f"Rejected for x0/x1 overlap: {n_rejected}. Total failed/excluded: {n_failed}")

# ------------------------------------------------------------
# Append to the existing double-sigmoid library
# ------------------------------------------------------------
existing = np.load(os.path.join(OUTPUT_DIR, "double_trajectories.npy"))
combined = np.vstack([existing, new_trajectories])
np.save(os.path.join(OUTPUT_DIR, "double_trajectories.npy"), combined)
np.save(os.path.join(OUTPUT_DIR, "double_trajectories_widened_x1_only.npy"), new_trajectories)  # kept separately too, for inspection

print(f"\nExpanded double_trajectories.npy: {existing.shape[0]} original + "
      f"{NUM_NEW_SAMPLES} widened-x1 candidates = {combined.shape[0]} total rows.")
print("single_trajectories.npy was not touched.")