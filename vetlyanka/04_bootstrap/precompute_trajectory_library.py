"""
precompute_trajectory_library.py   (step 04 -- run after 02_fits; before the bootstrap)

One-time precomputation for the parametric bootstrap test (sigmoidal pair).

For every candidate parameter set in the shared LHS file
(01_sampling/parameter_samples_<NUM_SAMPLES>.csv -- the same file the fits
used), this script solves the ODE once under the single-sigmoid model and
once under the double-sigmoid model, and stores only the 22 weekly
cumulative-death predictions for each.

WHY: simulating a parameter set does not depend on which dataset (real or
synthetic) it is later scored against. Solving every candidate once here
lets the bootstrap score thousands of synthetic datasets with plain array
arithmetic, and it searches the same candidate set as the real fits.

X0/X1 CONSTRAINT: double-sigmoid candidates with x1 - x0 < X0_X1_MIN_GAP are
excluded (left as NaN), exactly as in polish_vetlyanka_fit_double.py. This
does not apply to the single-sigmoid library (x1 is inert there).

T_PERCEIVE: in the real single-sigmoid fit T_perceive is FIXED at the
double fit's value, so it is fixed here too -- READ AUTOMATICALLY from
02_fits/results/double/vetlyanka_polished_fit.csv.

OBSERVATION TIMES: weeks 1..22 at t = 1..22 (same as the fits).

Re-run only if the sample file, the model files, X0_X1_MIN_GAP, the double
fit (its T_perceive), or the alignment change.

Writes to 04_bootstrap/results/sigmoidal/:
    double_trajectories.npy   (num_samples, 22)
    single_trajectories.npy   (num_samples, 22)
    candidate_params.npy      (num_samples, 13)  -- the raw sample rows
    library_info.json         -- settings used, so later steps can check them
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
import pandas as pd
from scipy.integrate import solve_ivp

from plague_double_sigmoid_model import plague_model as plague_model_double
from plague_single_sigmoid_model import plague_model as plague_model_single
from vetlyanka_bounds import PARAM_NAMES

# ---------------------------------------------------------------------
# Settings -- must match the 02_fits scripts
# ---------------------------------------------------------------------
LEGACY_ALIGNMENT = False   # True = published (t = 0..21); False = corrected (t = 1..22)

NUM_SAMPLES = 400000       # must match 01_sampling/parameter_samples_<NUM_SAMPLES>.csv
X0_X1_MIN_GAP = 1.0        # must match polish_vetlyanka_fit_double.py
MAX_SAMPLES = None         # e.g. 2000 for a quick smoke test; None for the full run

SAMPLES_PATH = os.path.join(ROOT, "01_sampling", f"parameter_samples_{NUM_SAMPLES}.csv")
DOUBLE_FIT_PATH = os.path.join(ROOT, "02_fits", "results", "double", "vetlyanka_polished_fit.csv")
OUTPUT_DIR = os.path.join(HERE, "results", "sigmoidal")
os.makedirs(OUTPUT_DIR, exist_ok=True)

t_start, t_end, dt = 0.0, 22.0, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
n_weeks = 22
OBS = slice(0, n_weeks) if LEGACY_ALIGNMENT else slice(1, n_weeks + 1)
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]  # S, I, R, DI, DDI, DR, P

IDX_T_PERCEIVE = PARAM_NAMES.index("T_perceive")
IDX_X0 = PARAM_NAMES.index("x0")
IDX_X1 = PARAM_NAMES.index("x1")


def load_t_perceive_from_double():
    if not os.path.exists(DOUBLE_FIT_PATH):
        raise FileNotFoundError(f"{DOUBLE_FIT_PATH} not found -- run 02_fits/polish_vetlyanka_fit_double.py first.")
    fit = pd.read_csv(DOUBLE_FIT_PATH)
    return float(fit.loc[fit["Parameter"] == "T_perceive", "Best_Fit"].iloc[0])


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
    T_PERCEIVE_FIXED_SIGMOID = load_t_perceive_from_double()
    if not os.path.exists(SAMPLES_PATH):
        raise FileNotFoundError(f"{SAMPLES_PATH} not found -- run 01_sampling/parameter_sampling.py first.")
    sampleparam = np.loadtxt(SAMPLES_PATH, delimiter=",", skiprows=1)
    if MAX_SAMPLES is not None:
        sampleparam = sampleparam[:MAX_SAMPLES]
    num_samples = sampleparam.shape[0]

    print(f"Observation times: {'t = 0..21 (LEGACY)' if LEGACY_ALIGNMENT else 't = 1..22 (end of each week)'}")
    print(f"Samples: {SAMPLES_PATH}  ({num_samples} rows)")
    print(f"Double: excluding candidates with x1 - x0 < {X0_X1_MIN_GAP} weeks")
    print(f"Single: T_perceive fixed at {T_PERCEIVE_FIXED_SIGMOID:.6f} (from the double fit)")
    print(f"Writing to: {OUTPUT_DIR}\n")

    double_trajectories = np.full((num_samples, n_weeks), np.nan)
    single_trajectories = np.full((num_samples, n_weeks), np.nan)
    n_rejected_double = 0
    t0 = time.time()

    for i, params in enumerate(sampleparam):
        if i % 10000 == 0 and i > 0:
            rate = i / (time.time() - t0)
            print(f"  sample {i}/{num_samples}  (double rejected so far: {n_rejected_double}; "
                  f"~{(num_samples - i) / rate / 60:.0f} min left)", flush=True)
            gc.collect()

        # Double: skip degenerate x0/x1 overlap (left as NaN; the bootstrap filters NaN rows)
        if params[IDX_X0] < params[IDX_X1] - X0_X1_MIN_GAP:
            w_double = simulate_weekly(plague_model_double, params)
            if w_double is not None:
                double_trajectories[i, :] = w_double
        else:
            n_rejected_double += 1

        # Single: x1 is inert; T_perceive fixed as in the real single fit
        params_single = params.copy()
        params_single[IDX_T_PERCEIVE] = T_PERCEIVE_FIXED_SIGMOID
        w_single = simulate_weekly(plague_model_single, params_single)
        if w_single is not None:
            single_trajectories[i, :] = w_single

    np.save(os.path.join(OUTPUT_DIR, "double_trajectories.npy"), double_trajectories)
    np.save(os.path.join(OUTPUT_DIR, "single_trajectories.npy"), single_trajectories)
    np.save(os.path.join(OUTPUT_DIR, "candidate_params.npy"), sampleparam)
    info = {"samples_file": os.path.basename(SAMPLES_PATH), "num_samples": int(num_samples),
            "legacy_alignment": LEGACY_ALIGNMENT, "x0_x1_min_gap": X0_X1_MIN_GAP,
            "T_perceive_fixed_single": T_PERCEIVE_FIXED_SIGMOID,
            "T_perceive_source": os.path.relpath(DOUBLE_FIT_PATH, ROOT)}
    with open(os.path.join(OUTPUT_DIR, "library_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)

    n_failed_double = int(np.isnan(double_trajectories).any(axis=1).sum())
    n_failed_single = int(np.isnan(single_trajectories).any(axis=1).sum())
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Saved to {OUTPUT_DIR}")
    print(f"Double -- rejected for x0/x1 overlap: {n_rejected_double}, total failed/excluded: {n_failed_double}")
    print(f"Single -- failed solves: {n_failed_single}")


if __name__ == "__main__":
    main()