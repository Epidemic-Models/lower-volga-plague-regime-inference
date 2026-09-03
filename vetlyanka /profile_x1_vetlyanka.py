# profile_x1_vetlyanka.py

"""
Profile-SSE analysis for the decline-onset parameter x1 in the
standalone Vetlyanka double-regime model.

PURPOSE
-------
Assess practical identifiability of x1 directly from the observed
Vetlyanka mortality trajectory.

For each fixed x1 value across its admissible range:
    1. Hold x1 fixed.
    2. Re-optimize all remaining model parameters.
    3. Use multiple starting points.
    4. Record the minimum achievable SSE.

The resulting profile

    x1 -> min SSE(x1)

shows whether other parameters can compensate when x1 is forced away
from its optimum.

IMPORTANT
---------
This is a profile-SSE analysis, not automatically a formal likelihood
confidence interval.

All remaining parameters MUST be re-optimized at every fixed x1.
Simply changing x1 while holding the other fitted parameters fixed
would only be a one-at-a-time sensitivity analysis.

OUTPUTS
-------
    x1_profile_sse.csv
    x1_profile_sse.png

Optionally:
    x1_profile_best_parameters.csv

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

import os

# ============================================================
# 1. MODEL-SPECIFIC IMPORTS
# ============================================================

from vetlyanka_bounds import PARAM_NAMES, BOUNDS
from vetlyanka_simulation import simulate_vetlyanka





# ============================================================
# 2. OBSERVED VETLYANKA DATA
# ============================================================

observed = np.array([
    3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19,
    27, 34, 90, 259, 313, 345, 364, 376,
    376, 376, 376
], dtype=float)


# ============================================================
# 3. SETTINGS
# ============================================================

# x1 admissible range from your table:
X1_MIN = 14.0
X1_MAX = 20.0

# 0.1-week resolution:
X1_GRID = np.arange(X1_MIN, X1_MAX + 1e-9, 0.1)

# Number of independently polished starts at each fixed x1.
N_STARTS = 20

# Reproducibility.
RANDOM_SEED = 12345

# L-BFGS-B settings.
MAXITER = 5000
FTOL = 1e-12
GTOL = 1e-8

# Large objective returned for invalid parameter combinations.
PENALTY = 1e30

# Use existing sampled candidates as starting points if available.
USE_CANDIDATE_LIBRARY = True



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATE_PARAMS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "bootstrap",
    "sigmoidal",
    "candidate_params.npy",
)

TRAJECTORIES_FILE = os.path.join(
    BASE_DIR,
    "data",
    "bootstrap",
    "sigmoidal",
    "double_trajectories.npy",
)


# Number of good candidate-library points considered when selecting
# starts close to each fixed x1.
CANDIDATE_POOL_SIZE = 500


# ============================================================
# 4. BASIC CHECKS
# ============================================================

if "x1" not in PARAM_NAMES:
    raise ValueError(
        f"'x1' not found in PARAM_NAMES:\n{PARAM_NAMES}"
    )

X1_INDEX = PARAM_NAMES.index("x1")

if len(PARAM_NAMES) != len(BOUNDS):
    raise ValueError(
        "PARAM_NAMES and BOUNDS must have the same length."
    )

N_PARAMS = len(PARAM_NAMES)

FREE_PARAM_NAMES = [
    name
    for i, name in enumerate(PARAM_NAMES)
    if i != X1_INDEX
]

FREE_BOUNDS = [
    bound
    for i, bound in enumerate(BOUNDS)
    if i != X1_INDEX
]

if len(FREE_PARAM_NAMES) != N_PARAMS - 1:
    raise RuntimeError("Incorrect number of free parameters.")


# ============================================================
# 5. PARAMETER VECTOR HELPERS
# ============================================================

def insert_fixed_x1(free_params, fixed_x1):
    """
    Reconstruct the full parameter vector from the free parameter
    vector while inserting fixed_x1 at the original x1 position.
    """
    free_params = np.asarray(free_params, dtype=float)

    if len(free_params) != N_PARAMS - 1:
        raise ValueError(
            f"Expected {N_PARAMS - 1} free parameters, "
            f"received {len(free_params)}."
        )

    full_params = np.empty(N_PARAMS, dtype=float)

    j = 0
    for i in range(N_PARAMS):
        if i == X1_INDEX:
            full_params[i] = fixed_x1
        else:
            full_params[i] = free_params[j]
            j += 1

    return full_params


def remove_x1(full_params):
    """
    Remove x1 from a complete parameter vector.
    """
    full_params = np.asarray(full_params, dtype=float)
    return np.delete(full_params, X1_INDEX)


# ============================================================
# 6. PARAMETER CONSTRAINTS
# ============================================================

def valid_full_parameter_vector(params):
    """
    Check constraints not automatically enforced by L-BFGS-B.

    At minimum, enforce x0 < x1.

    Add any other cross-parameter constraints used in your
    primary fitting code here.
    """
    params = np.asarray(params, dtype=float)

    param_dict = {
        name: params[i]
        for i, name in enumerate(PARAM_NAMES)
    }

    # Important double-sigmoid ordering constraint
    if "x0" in param_dict:
        if not (param_dict["x0"] < param_dict["x1"]):
            return False

    return True


# ============================================================
# 7. MODEL OUTPUT WRAPPER
# ============================================================

def get_model_trajectory(params):
    """
    Run the standalone Vetlyanka model and return the cumulative
    mortality trajectory at the same 22 observation points as
    `observed`.
    """
    try:
        prediction = simulate_vetlyanka(params)
    except Exception:
        return None

    if prediction is None:
        return None

    prediction = np.asarray(prediction, dtype=float).reshape(-1)

    if len(prediction) != len(observed):
        raise ValueError(
            "simulate_vetlyanka(params) returned "
            f"{len(prediction)} values, but observed has "
            f"{len(observed)} values."
        )

    if np.any(~np.isfinite(prediction)):
        return None

    return prediction


# ============================================================
# 8. PROFILE OBJECTIVE
# ============================================================

def profile_objective(free_params, fixed_x1):
    """
    SSE objective with x1 held fixed.
    """
    full_params = insert_fixed_x1(
        free_params,
        fixed_x1
    )

    if not valid_full_parameter_vector(full_params):
        return PENALTY

    prediction = get_model_trajectory(full_params)

    if prediction is None:
        return PENALTY

    residuals = prediction - observed
    sse = np.sum(residuals ** 2)

    if not np.isfinite(sse):
        return PENALTY

    return float(sse)


# ============================================================
# 9. RANDOM START GENERATION
# ============================================================

def random_free_start(rng, fixed_x1):
    """
    Draw one feasible free-parameter starting point uniformly
    from the box constraints.

    x0 is additionally constrained to be below fixed_x1.
    """
    start = np.empty(len(FREE_BOUNDS), dtype=float)

    for j, (name, bound) in enumerate(
        zip(FREE_PARAM_NAMES, FREE_BOUNDS)
    ):
        low, high = bound

        if name == "x0":
            high = min(
                high,
                fixed_x1 - 1e-5
            )

            if high <= low:
                return None

        start[j] = rng.uniform(low, high)

    return start


# ============================================================
# 10. EXISTING-LIBRARY STARTS
# ============================================================

def load_candidate_library():
    """
    Load the existing LHS parameter library and associated
    trajectories.

    Returns
    -------
    full_candidates : ndarray
    candidate_sse   : ndarray
    """
    candidate_params = np.load(CANDIDATE_PARAMS_FILE)
    trajectories = np.load(TRAJECTORIES_FILE)

    if candidate_params.shape[0] != trajectories.shape[0]:
        raise ValueError(
            "candidate_params.npy and double_trajectories.npy "
            "contain different numbers of candidates."
        )

    valid = (
        ~np.isnan(trajectories).any(axis=1)
        & np.isfinite(trajectories).all(axis=1)
    )

    candidate_params = candidate_params[valid]
    trajectories = trajectories[valid]

    if candidate_params.shape[1] != N_PARAMS:
        raise ValueError(
            "Candidate parameter dimension does not match "
            "PARAM_NAMES."
        )

    candidate_sse = np.sum(
        (trajectories - observed) ** 2,
        axis=1
    )

    return candidate_params, candidate_sse


def get_library_starts(
    fixed_x1,
    candidate_params,
    candidate_sse,
    n_starts
):
    """
    Select good LHS-library starting points for a given fixed x1.
    """
    order = np.argsort(candidate_sse)

    pool_n = min(
        CANDIDATE_POOL_SIZE,
        len(order)
    )

    pool_idx = order[:pool_n]
    pool = candidate_params[pool_idx]

    x1_distance = np.abs(
        pool[:, X1_INDEX] - fixed_x1
    )

    local_order = np.argsort(x1_distance)

    selected = []

    for idx in local_order:
        full = pool[idx].copy()
        full[X1_INDEX] = fixed_x1

        # Ensure x0 < fixed x1.
        if not valid_full_parameter_vector(full):
            continue

        selected.append(
            remove_x1(full)
        )

        if len(selected) >= n_starts:
            break

    return selected


# ============================================================
# 11. POLISH ONE FIXED x1 VALUE
# ============================================================

def optimize_at_fixed_x1(
    fixed_x1,
    rng,
    candidate_params=None,
    candidate_sse=None
):
    """
    Optimize all non-x1 parameters for one fixed x1.

    Returns
    -------
    best_sse
    best_full_params
    optimization_records
    """

    starts = []

    # A. Starts from existing LHS library
    if (
        USE_CANDIDATE_LIBRARY
        and candidate_params is not None
        and candidate_sse is not None
    ):
        library_starts = get_library_starts(
            fixed_x1=fixed_x1,
            candidate_params=candidate_params,
            candidate_sse=candidate_sse,
            n_starts=N_STARTS
        )

        starts.extend(library_starts)

    # B. Fill remaining starts with random feasible points
    while len(starts) < N_STARTS:
        start = random_free_start(
            rng,
            fixed_x1
        )

        if start is not None:
            starts.append(start)

    # Ensure exactly N_STARTS.
    starts = starts[:N_STARTS]

    best_sse = np.inf
    best_full_params = None

    records = []

    for start_idx, start in enumerate(
        starts,
        start=1
    ):
        result = minimize(
            profile_objective,
            x0=np.asarray(start),
            args=(fixed_x1,),
            method="L-BFGS-B",
            bounds=FREE_BOUNDS,
            options={
                "maxiter": MAXITER,
                "ftol": FTOL,
                "gtol": GTOL
            }
        )

        final_sse = float(result.fun)

        record = {
            "x1": fixed_x1,
            "start_index": start_idx,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "n_iterations": getattr(
                result,
                "nit",
                np.nan
            ),
            "n_function_evals": getattr(
                result,
                "nfev",
                np.nan
            ),
            "sse": final_sse
        }

        records.append(record)

        print(
            f"    start {start_idx:02d}/{N_STARTS}: "
            f"SSE = {final_sse:.6f}, "
            f"success = {result.success}"
        )

        if (
            np.isfinite(final_sse)
            and final_sse < best_sse
        ):
            best_sse = final_sse
            best_full_params = insert_fixed_x1(
                result.x,
                fixed_x1
            )

    return (
        best_sse,
        best_full_params,
        records
    )


# ============================================================
# 12. MAIN PROFILE LOOP
# ============================================================

def main():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    print("=" * 70)
    print("PROFILE-SSE ANALYSIS FOR x1")
    print("=" * 70)

    print(f"Number of parameters: {N_PARAMS}")
    print(
        f"Free parameters at each profile point: "
        f"{N_PARAMS - 1}"
    )
    print(f"x1 range: [{X1_MIN}, {X1_MAX}]")
    print(
        f"Number of x1 grid points: "
        f"{len(X1_GRID)}"
    )
    print(
        f"Optimization starts per x1: "
        f"{N_STARTS}"
    )

    # Load candidate library
    candidate_params = None
    candidate_sse = None

    if USE_CANDIDATE_LIBRARY:
        try:
            (
                candidate_params,
                candidate_sse
            ) = load_candidate_library()

            print("\nLoaded candidate library:")
            print(
                f"  valid candidates = "
                f"{len(candidate_sse)}"
            )
            print(
                f"  best raw-library SSE = "
                f"{candidate_sse.min():.6f}"
            )

        except Exception as exc:
            print(
                "\nWARNING: Could not load candidate "
                "library."
            )
            print(exc)
            print("Proceeding with random multi-starts.")

            candidate_params = None
            candidate_sse = None

    profile_rows = []
    all_optimization_records = []

    # Iterate over fixed x1
    for grid_idx, fixed_x1 in enumerate(
        X1_GRID,
        start=1
    ):
        print("\n" + "=" * 70)
        print(
            f"x1 = {fixed_x1:.3f} "
            f"({grid_idx}/{len(X1_GRID)})"
        )
        print("=" * 70)

        (
            best_sse,
            best_full_params,
            records
        ) = optimize_at_fixed_x1(
            fixed_x1=fixed_x1,
            rng=rng,
            candidate_params=candidate_params,
            candidate_sse=candidate_sse
        )

        all_optimization_records.extend(records)

        row = {
            "x1": fixed_x1,
            "profile_sse": best_sse
        }

        if best_full_params is not None:
            for name, value in zip(
                PARAM_NAMES,
                best_full_params
            ):
                row[name] = value
        else:
            for name in PARAM_NAMES:
                row[name] = np.nan

        profile_rows.append(row)

        # Save after every x1 so work survives interruption.
        pd.DataFrame(
            profile_rows
        ).to_csv(
            "x1_profile_sse_partial.csv",
            index=False
        )

        pd.DataFrame(
            all_optimization_records
        ).to_csv(
            "x1_profile_optimization_log_partial.csv",
            index=False
        )

    # Final profile table
    profile_df = pd.DataFrame(profile_rows)

    finite_mask = np.isfinite(
        profile_df["profile_sse"]
    )

    if not finite_mask.any():
        raise RuntimeError(
            "No finite profile SSE values were obtained."
        )

    profile_min = profile_df.loc[
        finite_mask,
        "profile_sse"
    ].min()

    profile_df["delta_sse"] = (
        profile_df["profile_sse"]
        - profile_min
    )

    best_idx = profile_df[
        "profile_sse"
    ].idxmin()

    profile_best_x1 = profile_df.loc[
        best_idx,
        "x1"
    ]

    profile_best_sse = profile_df.loc[
        best_idx,
        "profile_sse"
    ]

    profile_df.to_csv(
        "x1_profile_sse.csv",
        index=False
    )

    # Save best parameters at each x1
    parameter_columns = [
        "x1",
        "profile_sse",
        "delta_sse"
    ] + PARAM_NAMES

    existing_columns = [
        col
        for col in parameter_columns
        if col in profile_df.columns
    ]

    profile_df[
        existing_columns
    ].to_csv(
        "x1_profile_best_parameters.csv",
        index=False
    )

    # Save optimization log
    optimization_log_df = pd.DataFrame(
        all_optimization_records
    )

    optimization_log_df.to_csv(
        "x1_profile_optimization_log.csv",
        index=False
    )

    # PROFILE PLOT
    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        profile_df["x1"],
        profile_df["delta_sse"],
        marker="o",
        markersize=3
    )

    plt.axvline(
        profile_best_x1,
        linestyle="--",
        label=(
            rf"Profile minimum "
            rf"$x_1={profile_best_x1:.2f}$"
        )
    )

    plt.xlabel(
        r"Fixed decline-onset parameter $x_1$ (weeks)"
    )

    plt.ylabel(
        r"$\Delta$SSE relative to profile minimum"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "x1_profile_sse.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # PRINT SUMMARY
    print("\n" + "=" * 70)
    print("PROFILE COMPLETE")
    print("=" * 70)

    print(
        f"Profile minimum SSE: "
        f"{profile_best_sse:.6f}"
    )

    print(
        f"Profile minimum x1: "
        f"{profile_best_x1:.3f} weeks"
    )

    print("\nLowest profile points:")
    print(
        profile_df[
            ["x1", "profile_sse", "delta_sse"]
        ]
        .sort_values("profile_sse")
        .head(10)
        .to_string(index=False)
    )

    print("\nSaved:")
    print("  x1_profile_sse.csv")
    print("  x1_profile_best_parameters.csv")
    print("  x1_profile_optimization_log.csv")
    print("  x1_profile_sse.png")


# ============================================================
# 18. RUN
# ============================================================

if __name__ == "__main__":
    main()
