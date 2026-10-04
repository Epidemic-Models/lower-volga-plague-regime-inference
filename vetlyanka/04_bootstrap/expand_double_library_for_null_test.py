"""
expand_double_library_for_null_test.py   (step 04 -- run AFTER precompute_trajectory_library.py)

The double-sigmoid library from precompute_trajectory_library.py is built
under the real-fit bounds, so it can only represent curves whose decline
(x1) falls inside those bounds -- it cannot represent "no decline inside the
22-week window". The parametric bootstrap needs the double-sigmoid candidate
pool to have a genuine chance of representing "no second regime" when that
is true, or the null test is biased. This script adds a supplementary batch
with x1 widened to 40 weeks (well past the observation window).

X0/X1 CONSTRAINT: candidates with x1 - x0 < X0_X1_MIN_GAP are excluded, as in
polish_vetlyanka_fit_double.py. gamma/mu bounds are left untouched
(historically grounded).

SAMPLING: scipy.stats.qmc.LatinHypercube(seed=LHS_SEED) -- reproducible.
(The previous version used pyDOE with no seed, so every run differed.)

SAFE TO RE-RUN: the main library rows are taken from the first
num_samples rows of double_trajectories.npy (num_samples is read from
library_info.json written by precompute_trajectory_library.py), so running
this script twice does NOT append the widened batch twice.

OBSERVATION TIMES: weeks 1..22 at t = 1..22 (same as the fits).

Reads/writes 04_bootstrap/results/sigmoidal/:
    double_trajectories.npy                 -> main rows + widened rows
    double_trajectories_widened_x1_only.npy -> widened rows only (inspection)
    candidate_params_widened_x1.npy         -> widened parameter rows (13 columns);
                                               row i = library row num_samples + i
    library_info.json                       -> updated with the widened-batch settings
single_trajectories.npy is not touched.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # .../vetlyanka/04_bootstrap
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):  # walk up to vetlyanka/, which contains src/
    if os.path.dirname(ROOT) == ROOT:
        raise FileNotFoundError("Could not find the src/ folder above this script.")
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import gc
import json
import time
import numpy as np
from scipy.stats import qmc
from scipy.integrate import solve_ivp

from plague_double_sigmoid_model import plague_model as plague_model_double
from vetlyanka_bounds import PARAMETERS, PARAM_NAMES

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
LEGACY_ALIGNMENT = False   # must match precompute_trajectory_library.py
NUM_NEW_SAMPLES = 100000   # 1:4 ratio to the 400,000-row main library (same ratio as the frozen library)
LHS_SEED = 43              # fixed seed -> same widened batch every run
X0_X1_MIN_GAP = 1.0        # must match polish_vetlyanka_fit_double.py
X1_WIDENED_UPPER = 40.0

LIB_DIR = os.path.join(HERE, "results", "sigmoidal")
INFO_PATH = os.path.join(LIB_DIR, "library_info.json")

# Widened bounds: built FROM the shared PARAMETERS, only x1's upper bound overridden
PARAMETERS_WIDENED = [(name, lo, X1_WIDENED_UPPER if name == "x1" else hi) for name, lo, hi in PARAMETERS]
IDX_X0 = PARAM_NAMES.index("x0")
IDX_X1 = PARAM_NAMES.index("x1")

t_start, t_end, dt = 0.0, 22.0, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
n_weeks = 22
OBS = slice(0, n_weeks) if LEGACY_ALIGNMENT else slice(1, n_weeks + 1)
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]


def simulate_weekly(model_fn, params):
    """22 weekly cumulative deaths (DR) at the observation times, or None on failure."""
    try:
        sol = solve_ivp(model_fn, [t_start, t_end], initial_conditions,
                        args=(params,), t_eval=t_points, method="RK45")
    except (ValueError, FloatingPointError, OverflowError):
        return None
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    weekly = sol.y[5][::steps_per_week][OBS]
    return weekly if len(weekly) == n_weeks else None


def main():
    if not os.path.exists(INFO_PATH):
        raise FileNotFoundError(f"{INFO_PATH} not found -- run precompute_trajectory_library.py first.")
    with open(INFO_PATH) as fh:
        info = json.load(fh)
    if info.get("legacy_alignment") != LEGACY_ALIGNMENT:
        raise ValueError("LEGACY_ALIGNMENT here does not match the main library -- set them the same.")
    n_main = int(info["num_samples"])

    existing = np.load(os.path.join(LIB_DIR, "double_trajectories.npy"))
    if existing.shape[0] < n_main:
        raise ValueError(f"double_trajectories.npy has {existing.shape[0]} rows, expected at least {n_main}.")
    main_rows = existing[:n_main]   # drops any widened rows from an earlier run of this script

    lo = np.array([p[1] for p in PARAMETERS_WIDENED], dtype=float)
    hi = np.array([p[2] for p in PARAMETERS_WIDENED], dtype=float)
    unit = qmc.LatinHypercube(d=len(PARAMETERS_WIDENED), seed=LHS_SEED).random(n=NUM_NEW_SAMPLES)
    new_samples = lo + unit * (hi - lo)

    print(f"Observation times: {'t = 0..21 (LEGACY)' if LEGACY_ALIGNMENT else 't = 1..22 (end of each week)'}")
    print(f"Main library: {n_main} rows. Simulating {NUM_NEW_SAMPLES} widened-x1 candidates "
          f"(x1 up to {X1_WIDENED_UPPER:g}; excluding x1 - x0 < {X0_X1_MIN_GAP})...")

    new_trajectories = np.full((NUM_NEW_SAMPLES, n_weeks), np.nan)
    n_rejected = 0
    t0 = time.time()
    for i, params in enumerate(new_samples):
        if i % 10000 == 0 and i > 0:
            rate = i / (time.time() - t0)
            print(f"  sample {i}/{NUM_NEW_SAMPLES}  (rejected so far: {n_rejected}; "
                  f"~{(NUM_NEW_SAMPLES - i) / rate / 60:.0f} min left)", flush=True)
            gc.collect()
        if params[IDX_X0] >= params[IDX_X1] - X0_X1_MIN_GAP:
            n_rejected += 1
            continue
        w = simulate_weekly(plague_model_double, params)
        if w is not None:
            new_trajectories[i, :] = w

    n_failed = int(np.isnan(new_trajectories).any(axis=1).sum())
    print(f"Rejected for x0/x1 overlap: {n_rejected}. Total failed/excluded: {n_failed}")

    combined = np.vstack([main_rows, new_trajectories])
    np.save(os.path.join(LIB_DIR, "double_trajectories.npy"), combined)
    np.save(os.path.join(LIB_DIR, "double_trajectories_widened_x1_only.npy"), new_trajectories)
    np.save(os.path.join(LIB_DIR, "candidate_params_widened_x1.npy"), new_samples)

    info.update({"widened_x1_appended": True, "num_widened_samples": NUM_NEW_SAMPLES,
                 "widened_lhs_seed": LHS_SEED, "widened_x1_upper": X1_WIDENED_UPPER,
                 "double_library_rows": int(combined.shape[0])})
    with open(INFO_PATH, "w") as fh:
        json.dump(info, fh, indent=2)

    print(f"\nExpanded double_trajectories.npy: {n_main} main + {NUM_NEW_SAMPLES} widened-x1 "
          f"= {combined.shape[0]} rows. single_trajectories.npy was not touched.")


if __name__ == "__main__":
    main()