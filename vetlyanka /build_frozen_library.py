"""
build_frozen_library.py

Builds the precomputed trajectory library needed for the frozen-model
(double-frozen vs single-frozen) parametric bootstrap and confidence-
interval scripts, mirroring the original pipeline (parameter_sampling.py
-> precompute_trajectory_library.py -> expand_double_library_for_null_test.py)
but adapted for the reduced (constant-gamma/mu) parameter spaces.

Key idea, same as the original: LHS-sample the DOUBLE-frozen model's own
11-dimensional reduced space (it's the superset -- includes x1, c1, which
single-frozen doesn't have), expand each candidate to the full 13-slot
vector, then evaluate that SAME candidate under both plague_model_double
and plague_model_single. Single-sigmoid structurally ignores x1/c1, so
this gives a fair, shared candidate pool for both models' trajectory
libraries, exactly the trick the original double/single bootstrap used.

X0/X1 ORDERING CONSTRAINT: candidates where the rise-midpoint (x0) isn't
meaningfully before the decline-onset (x1) are excluded from the DOUBLE
trajectory library only, matching the same constraint now enforced in
polish_vetlyanka_fit_double_freeze_mu_gamma_FIXED.py -- without it, the
double-frozen library (and any CI computed from it) could include
degenerate candidates where the two sigmoids overlap into a non-physical
shape that still happens to fit well. This constraint does NOT apply to
the single-frozen trajectory: x1 is completely inert in the single-
sigmoid model, so a candidate failing this check is still a perfectly
valid, independent draw of the parameters single-frozen actually uses --
rejecting it there would only reduce single-frozen's effective sample
density for no reason.


T_PERCEIVE OVERRIDE FOR SINGLE: the real single-frozen fit
(polish_vetlyanka_fit_single_freeze_mu_gamma_FIXED.py) FIXES T_perceive
at 3.409, the converged value from the double-frozen model in the same
constant-gamma/mu family. This library must represent the actual fitting
procedure being tested downstream (bootstrap, CI), so every single-
sigmoid trajectory is computed with T_perceive forced to that same fixed
value, regardless of whatever value the shared LHS draw assigned it for
that row. The double-sigmoid trajectory for the same row is unaffected:
double-frozen's T_perceive is genuinely fitted and therefore remains
freely drawn.


A widened-x1 supplementary batch is appended to the double-frozen library
afterward, for the same reason as the original: the double-frozen library,
built under bounds appropriate for fitting the real data, can't represent
"no real decline within the window" candidates on its own, which biases
the null test against the null hypothesis. Pushing x1 out further gives
the double-frozen pool a genuine chance to represent "no second regime"
too.

Outputs:
    double_frozen_trajectories.npy   (num_samples, 22)
    single_frozen_trajectories.npy   (num_samples, 22)
    candidate_params_frozen_main.npy (num_samples, 11) -- for reference
"""

import numpy as np
from scipy.stats import qmc
from scipy.integrate import solve_ivp
import gc

from plague_double_sigmoid_model import plague_model as plague_model_double
from plague_single_sigmoid_model import plague_model as plague_model_single
from vetlyanka_bounds import BOUNDS

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "bootstrap", "frozen")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Settings -- must match the frozen fitting scripts exactly
# ------------------------------------------------------------
t_start, t_end, dt = 0, 22, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
n_weeks = 22
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]

NUM_MAIN_SAMPLES = 400000   # matched to the fitting scripts' sample count now that
                             # the bound fix below widens the search space
NUM_WIDENED_SAMPLES = 100000  # scaled up proportionally with NUM_MAIN_SAMPLES
LHS_SEED = 42
X0_X1_MIN_GAP = 2.0  # must match polish_vetlyanka_fit_double_freeze_mu_gamma_FIXED.py
T_PERCEIVE_FIXED_SINGLE = 3.409  # must match polish_vetlyanka_fit_single_freeze_mu_gamma_FIXED.py
IDX_T_PERCEIVE_FULL = 10  # T_perceive's index in the full 13-slot parameter vector

# Real double-sigmoid indices: b1,b2,x0,x1,c,c1,gamma1,gamma2,mu1,mu2,T_perceive,sensitivity,dispose_rate
IDX_GAMMA1, IDX_GAMMA2, IDX_MU1, IDX_MU2 = 6, 7, 8, 9

# FIXED: span (union of both original ranges), not average of their endpoints.
# The averaged version produced a gamma_const range with ZERO overlap with
# either gamma1 or gamma2's own range, and shrank mu_const's ceiling well
# below mu2's real, historically-grounded ceiling -- this must match exactly
# what fit_freeze_mu_gamma.py and the single-frozen fitting script now use.
GAMMA_CONST_BOUND = (min(BOUNDS[IDX_GAMMA1][0], BOUNDS[IDX_GAMMA2][0]),
                      max(BOUNDS[IDX_GAMMA1][1], BOUNDS[IDX_GAMMA2][1]))
MU_CONST_BOUND = (min(BOUNDS[IDX_MU1][0], BOUNDS[IDX_MU2][0]),
                   max(BOUNDS[IDX_MU1][1], BOUNDS[IDX_MU2][1]))

# Reduced double-frozen space: b1,b2,x0,x1,c,c1,gamma_const,mu_const,T_perceive,sensitivity,dispose_rate
REDUCED_NAMES = ["b1", "b2", "x0", "x1", "c", "c1", "gamma_const", "mu_const",
                  "T_perceive", "sensitivity", "dispose_rate"]
REDUCED_BOUNDS = [BOUNDS[0], BOUNDS[1], BOUNDS[2], BOUNDS[3], BOUNDS[4], BOUNDS[5],
                   GAMMA_CONST_BOUND, MU_CONST_BOUND, BOUNDS[10], BOUNDS[11], BOUNDS[12]]
K_REDUCED = len(REDUCED_NAMES)
IDX_X0_REDUCED = REDUCED_NAMES.index("x0")
IDX_X1_REDUCED = REDUCED_NAMES.index("x1")

# Same widened-x1 override as the original expand_double_library_for_null_test.py
WIDENED_BOUNDS = [(lo, 40 if name == "x1" else hi) for name, (lo, hi) in zip(REDUCED_NAMES, REDUCED_BOUNDS)]


def reduced_to_full(reduced_params):
    """Expand an 11-slot double-frozen vector to the full 13-slot vector."""
    (b1, b2, x0, x1, c, c1, gamma_const, mu_const,
     T_perceive, sensitivity, dispose_rate) = reduced_params
    return np.array([b1, b2, x0, x1, c, c1, gamma_const, gamma_const,
                      mu_const, mu_const, T_perceive, sensitivity, dispose_rate])


def simulate_weekly(model_fn, full_params):
    sol = solve_ivp(model_fn, [t_start, t_end], initial_conditions,
                     args=(full_params,), t_eval=t_points, method="RK45")
    if not sol.success:
        return None
    DR = sol.y[5]
    weekly = DR[::steps_per_week][:n_weeks]
    if len(weekly) < n_weeks:
        return None
    return weekly


def draw_lhs(bounds, n_samples, seed):
    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
    unit = sampler.random(n=n_samples)
    return np.array([[row[j] * (bounds[j][1] - bounds[j][0]) + bounds[j][0]
                       for j in range(len(bounds))] for row in unit])


def build_library(reduced_samples, label):
    n = reduced_samples.shape[0]
    double_traj = np.full((n, n_weeks), np.nan)
    single_traj = np.full((n, n_weeks), np.nan)
    n_rejected_double = 0
    for i, reduced in enumerate(reduced_samples):
        if i % 20000 == 0 and i > 0:
            print(f"  [{label}] sample {i}/{n}  (double rejected so far: {n_rejected_double})")
            gc.collect()
        full = reduced_to_full(reduced)

        # Double: skip candidates where x0/x1 overlap into the degenerate
        # configuration -- see module docstring. Leaves double_traj[i]
        # as NaN, which every downstream script already filters out.
        # T_perceive is genuinely fitted in the real double-frozen model,
        # so it stays as freely drawn here.
        x0, x1 = reduced[IDX_X0_REDUCED], reduced[IDX_X1_REDUCED]
        if x0 < x1 - X0_X1_MIN_GAP:
            w_d = simulate_weekly(plague_model_double, full)
            if w_d is not None:
                double_traj[i, :] = w_d
        else:
            n_rejected_double += 1

        # Single: x1 is inert here, so this candidate is still a valid,
        # independent draw of the parameters single-frozen actually uses --
        # always compute it regardless of the double-only rejection above.
        # T_perceive is FIXED in the real single-frozen fit, not searched --
        # override it here too, so this library represents the actual
        # model being CI'd/bootstrapped, not a more flexible one.
        full_single = full.copy()
        full_single[IDX_T_PERCEIVE_FULL] = T_PERCEIVE_FIXED_SINGLE
        w_s = simulate_weekly(plague_model_single, full_single)
        if w_s is not None:
            single_traj[i, :] = w_s

    print(f"  [{label}] done. Double rejected for x0/x1 overlap: {n_rejected_double}/{n}")
    return double_traj, single_traj


if __name__ == "__main__":
    print(f"Building main library ({NUM_MAIN_SAMPLES} samples)...")
    print(f"Single-frozen T_perceive forced to {T_PERCEIVE_FIXED_SINGLE} for every candidate.")
    main_samples = draw_lhs(REDUCED_BOUNDS, NUM_MAIN_SAMPLES, LHS_SEED)
    double_traj_main, single_traj_main = build_library(main_samples, "main")

    n_failed_d = int(np.isnan(double_traj_main).any(axis=1).sum())
    n_failed_s = int(np.isnan(single_traj_main).any(axis=1).sum())
    print(f"Main library done. Failed/excluded -- double: {n_failed_d}, single: {n_failed_s}")

    print(f"\nBuilding widened-x1 supplementary batch ({NUM_WIDENED_SAMPLES} samples)...")
    widened_samples = draw_lhs(WIDENED_BOUNDS, NUM_WIDENED_SAMPLES, LHS_SEED + 1)
    double_traj_widened, _ = build_library(widened_samples, "widened")
    n_failed_w = int(np.isnan(double_traj_widened).any(axis=1).sum())
    print(f"Widened batch done. Failed/excluded: {n_failed_w}")

    double_traj_combined = np.vstack([double_traj_main, double_traj_widened])

    np.save(os.path.join(OUTPUT_DIR, "double_frozen_trajectories.npy"), double_traj_combined)
    np.save(os.path.join(OUTPUT_DIR, "single_frozen_trajectories.npy"), single_traj_main)
    np.save(os.path.join(OUTPUT_DIR, "candidate_params_frozen_main.npy"), main_samples)

    print(f"\nSaved {OUTPUT_DIR}/double_frozen_trajectories.npy: {double_traj_main.shape[0]} main + "
          f"{double_traj_widened.shape[0]} widened-x1 = {double_traj_combined.shape[0]} total rows")
    print(f"Saved {OUTPUT_DIR}/single_frozen_trajectories.npy: {single_traj_main.shape[0]} rows")