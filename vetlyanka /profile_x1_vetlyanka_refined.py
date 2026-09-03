"""
profile_x1_vetlyanka_refined.py

Refined profile-SSE analysis for the decline-onset parameter x1
in the standalone Vetlyanka double-regime model.

For each fixed x1:
    - x1 is held fixed;
    - all other 12 parameters are re-optimized;
    - the unrestricted polished fit is used as a starting point;
    - good LHS candidates are used as starting points;
    - the best neighboring profile solution is used as a warm start;
    - the minimum SSE is retained.

This is a PROFILE-SSE analysis, not automatically a formal confidence
interval.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from vetlyanka_bounds import PARAM_NAMES, BOUNDS
from vetlyanka_simulation import simulate_vetlyanka


# ============================================================
# 1. OBSERVED VETLYANKA DATA
# ============================================================

OBSERVED = np.array([
    3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19,
    27, 34, 90, 259, 313, 345, 364, 376,
    376, 376, 376
], dtype=float)


# ============================================================
# 2. FILE PATHS
# ============================================================

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

POLISHED_FIT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "fits",
    "double",
    "vetlyanka_polished_fit.csv",  # <-- actual filename here
)


# ============================================================
# 3. PROFILE SETTINGS
# ============================================================

X1_INDEX = PARAM_NAMES.index("x1")

X1_MIN = BOUNDS[X1_INDEX][0]
X1_MAX = BOUNDS[X1_INDEX][1]

GRID_STEP = 0.1

X1_GRID = np.round(
    np.arange(
        X1_MIN,
        X1_MAX + GRID_STEP / 2,
        GRID_STEP
    ),
    10
)

N_STARTS = 30
N_LIBRARY_STARTS = 15

RANDOM_SEED = 12345

MAXITER = 5000
FTOL = 1e-12
GTOL = 1e-8

PENALTY = 1e30

CANDIDATE_POOL_SIZE = 2000

X0_X1_EPS = 1e-5


# ============================================================
# 4. FREE PARAMETER INFORMATION
# ============================================================

FREE_PARAM_NAMES = [
    name
    for i, name in enumerate(PARAM_NAMES)
    if i != X1_INDEX
]

FREE_BOUNDS_BASE = [
    bound
    for i, bound in enumerate(BOUNDS)
    if i != X1_INDEX
]


# ============================================================
# 5. PARAMETER VECTOR UTILITIES
# ============================================================

def insert_x1(free_params, fixed_x1):
    """
    Reconstruct full 13-parameter vector after removing x1
    from the optimizer.
    """

    free_params = np.asarray(
        free_params,
        dtype=float
    )

    full = np.empty(
        len(PARAM_NAMES),
        dtype=float
    )

    j = 0

    for i in range(len(PARAM_NAMES)):
        if i == X1_INDEX:
            full[i] = fixed_x1
        else:
            full[i] = free_params[j]
            j += 1

    return full


def remove_x1(full_params):
    """
    Remove x1 from a complete parameter vector.
    """

    return np.delete(
        np.asarray(full_params, dtype=float),
        X1_INDEX
    )


# ============================================================
# 6. PROFILE-SPECIFIC BOUNDS
# ============================================================

def get_profile_bounds(fixed_x1):
    """
    Construct bounds for the 12 free parameters.

    Enforce x0 < x1 directly in the optimizer bounds.
    """

    profile_bounds = []

    for name, (low, high) in zip(
        FREE_PARAM_NAMES,
        FREE_BOUNDS_BASE
    ):

        if name == "x0":
            high = min(
                high,
                fixed_x1 - X0_X1_EPS
            )

            if high <= low:
                raise ValueError(
                    f"No feasible x0 range when x1={fixed_x1}"
                )

        profile_bounds.append(
            (low, high)
        )

    return profile_bounds


# ============================================================
# 7. PARAMETER VALIDITY
# ============================================================

def parameter_vector_is_valid(params):

    p = dict(
        zip(PARAM_NAMES, params)
    )

    if p["x0"] >= p["x1"]:
        return False

    return True


# ============================================================
# 8. SSE OBJECTIVE
# ============================================================

def calculate_sse(full_params):

    if not parameter_vector_is_valid(
        full_params
    ):
        return PENALTY

    try:
        prediction = simulate_vetlyanka(
            full_params
        )
    except Exception:
        return PENALTY

    if prediction is None:
        return PENALTY

    prediction = np.asarray(
        prediction,
        dtype=float
    )

    if prediction.shape != OBSERVED.shape:
        return PENALTY

    if not np.all(
        np.isfinite(prediction)
    ):
        return PENALTY

    residuals = (
        prediction - OBSERVED
    )

    sse = np.sum(
        residuals ** 2
    )

    if not np.isfinite(sse):
        return PENALTY

    return float(sse)


def profile_objective(
    free_params,
    fixed_x1
):

    full_params = insert_x1(
        free_params,
        fixed_x1
    )

    return calculate_sse(
        full_params
    )


# ============================================================
# 9. LOAD POLISHED UNRESTRICTED FIT
# ============================================================

def load_polished_fit():

    print(
        f"\nLoading polished fit from:\n"
        f"{POLISHED_FIT_FILE}"
    )

    if not os.path.exists(
        POLISHED_FIT_FILE
    ):
        raise FileNotFoundError(
            f"Could not find polished fit:\n"
            f"{POLISHED_FIT_FILE}"
        )

    fit_df = pd.read_csv(
        POLISHED_FIT_FILE
    )

    if "Parameter" not in fit_df.columns:
        raise ValueError(
            "vetlyanka_polished_fit.csv must contain "
            "a column named 'Parameter'."
        )

    if "Best_Fit" not in fit_df.columns:
        raise ValueError(
            "vetlyanka_polished_fit.csv must contain "
            "a column named 'Best_Fit'."
        )

    fit_df = fit_df.set_index(
        "Parameter"
    )

    missing = [
        name
        for name in PARAM_NAMES
        if name not in fit_df.index
    ]

    if missing:
        raise ValueError(
            f"Missing parameters in polished fit: "
            f"{missing}"
        )

    polished = np.array(
        [
            fit_df.loc[
                name,
                "Best_Fit"
            ]
            for name in PARAM_NAMES
        ],
        dtype=float
    )

    return polished


# ============================================================
# 10. LOAD EXISTING LHS LIBRARY
# ============================================================

def load_existing_library():

    print(
        f"\nLoading candidate parameters from:\n"
        f"{CANDIDATE_PARAMS_FILE}"
    )

    print(
        f"\nLoading trajectories from:\n"
        f"{TRAJECTORIES_FILE}"
    )

    if not os.path.exists(
        CANDIDATE_PARAMS_FILE
    ):
        raise FileNotFoundError(
            f"Candidate parameter file not found:\n"
            f"{CANDIDATE_PARAMS_FILE}"
        )

    if not os.path.exists(
        TRAJECTORIES_FILE
    ):
        raise FileNotFoundError(
            f"Trajectory file not found:\n"
            f"{TRAJECTORIES_FILE}"
        )

    candidate_params = np.load(
        CANDIDATE_PARAMS_FILE
    )

    trajectories = np.load(
        TRAJECTORIES_FILE
    )

    if (
        candidate_params.shape[0]
        != trajectories.shape[0]
    ):
        raise ValueError(
            "candidate_params.npy and "
            "double_trajectories.npy have "
            "different row counts."
        )

    if (
        candidate_params.shape[1]
        != len(PARAM_NAMES)
    ):
        raise ValueError(
            "candidate_params.npy does not "
            "match the parameter dimension."
        )

    valid = np.isfinite(
        trajectories
    ).all(axis=1)

    candidate_params = (
        candidate_params[valid]
    )

    trajectories = (
        trajectories[valid]
    )

    candidate_sse = np.sum(
        (
            trajectories
            - OBSERVED
        ) ** 2,
        axis=1
    )

    return (
        candidate_params,
        candidate_sse
    )


# ============================================================
# 11. CLIP START INTO PROFILE BOUNDS
# ============================================================

def clip_to_profile_bounds(
    free_params,
    profile_bounds
):

    free_params = np.asarray(
        free_params,
        dtype=float
    ).copy()

    for i, (
        low,
        high
    ) in enumerate(
        profile_bounds
    ):

        free_params[i] = np.clip(
            free_params[i],
            low,
            high
        )

    return free_params


# ============================================================
# 12. RANDOM START
# ============================================================

def random_start(
    rng,
    profile_bounds
):

    return np.array(
        [
            rng.uniform(
                low,
                high
            )
            for low, high
            in profile_bounds
        ],
        dtype=float
    )


# ============================================================
# 13. GOOD LHS STARTS
# ============================================================

def get_library_starts(
    fixed_x1,
    candidate_params,
    candidate_sse,
    profile_bounds
):

    order = np.argsort(
        candidate_sse
    )

    pool_n = min(
        CANDIDATE_POOL_SIZE,
        len(order)
    )

    pool = candidate_params[
        order[:pool_n]
    ]

    distance = np.abs(
        pool[:, X1_INDEX]
        - fixed_x1
    )

    local_order = np.argsort(
        distance
    )

    starts = []

    for idx in local_order:

        full = pool[idx].copy()

        full[X1_INDEX] = fixed_x1

        free = remove_x1(
            full
        )

        free = clip_to_profile_bounds(
            free,
            profile_bounds
        )

        full_check = insert_x1(
            free,
            fixed_x1
        )

        if not parameter_vector_is_valid(
            full_check
        ):
            continue

        starts.append(
            free
        )

        if (
            len(starts)
            >= N_LIBRARY_STARTS
        ):
            break

    return starts


# ============================================================
# 14. REMOVE DUPLICATE STARTS
# ============================================================

def unique_starts(starts):

    unique = []

    for start in starts:

        start = np.asarray(
            start,
            dtype=float
        )

        duplicate = False

        for existing in unique:

            if np.allclose(
                start,
                existing,
                rtol=0,
                atol=1e-10
            ):
                duplicate = True
                break

        if not duplicate:
            unique.append(
                start
            )

    return unique


# ============================================================
# 15. BUILD START SET
# ============================================================

def make_starts(
    fixed_x1,
    polished_fit,
    candidate_params,
    candidate_sse,
    rng,
    warm_start=None
):

    profile_bounds = get_profile_bounds(
        fixed_x1
    )

    starts = []

    # --------------------------------------------------------
    # A. Warm continuation start
    # --------------------------------------------------------

    if warm_start is not None:

        warm_full = np.asarray(
            warm_start,
            dtype=float
        ).copy()

        warm_full[X1_INDEX] = (
            fixed_x1
        )

        warm_free = remove_x1(
            warm_full
        )

        warm_free = clip_to_profile_bounds(
            warm_free,
            profile_bounds
        )

        starts.append(
            warm_free
        )

    # --------------------------------------------------------
    # B. Unrestricted polished fit
    # --------------------------------------------------------

    polished_full = (
        polished_fit.copy()
    )

    polished_full[
        X1_INDEX
    ] = fixed_x1

    polished_free = remove_x1(
        polished_full
    )

    polished_free = clip_to_profile_bounds(
        polished_free,
        profile_bounds
    )

    starts.append(
        polished_free
    )

    # --------------------------------------------------------
    # C. LHS starts
    # --------------------------------------------------------

    lhs_starts = get_library_starts(
        fixed_x1=fixed_x1,
        candidate_params=candidate_params,
        candidate_sse=candidate_sse,
        profile_bounds=profile_bounds
    )

    starts.extend(
        lhs_starts
    )

    starts = unique_starts(
        starts
    )

    # --------------------------------------------------------
    # D. Random starts
    # --------------------------------------------------------

    while len(starts) < N_STARTS:

        starts.append(
            random_start(
                rng,
                profile_bounds
            )
        )

        starts = unique_starts(
            starts
        )

    return (
        starts[:N_STARTS],
        profile_bounds
    )


# ============================================================
# 16. OPTIMIZE ONE FIXED x1
# ============================================================

def optimize_fixed_x1(
    fixed_x1,
    polished_fit,
    candidate_params,
    candidate_sse,
    rng,
    warm_start=None
):

    (
        starts,
        profile_bounds
    ) = make_starts(
        fixed_x1=fixed_x1,
        polished_fit=polished_fit,
        candidate_params=candidate_params,
        candidate_sse=candidate_sse,
        rng=rng,
        warm_start=warm_start
    )

    best_sse = np.inf
    best_params = None

    records = []

    for start_number, start in enumerate(
        starts,
        start=1
    ):

        result = minimize(
            fun=profile_objective,
            x0=start,
            args=(fixed_x1,),
            method="L-BFGS-B",
            bounds=profile_bounds,
            options={
                "maxiter": MAXITER,
                "ftol": FTOL,
                "gtol": GTOL
            }
        )

        final_sse = float(
            result.fun
        )

        valid_result = (
            np.isfinite(
                final_sse
            )
            and
            final_sse
            < PENALTY / 10
        )

        print(
            f"    start "
            f"{start_number:02d}/{N_STARTS}: "
            f"SSE={final_sse:.6f}, "
            f"success={result.success}"
        )

        records.append(
            {
                "x1": fixed_x1,
                "start": start_number,
                "SSE": final_sse,
                "success": bool(
                    result.success
                ),
                "status": int(
                    result.status
                ),
                "message": str(
                    result.message
                ),
                "valid_result": bool(
                    valid_result
                )
            }
        )

        if (
            valid_result
            and
            final_sse < best_sse
        ):

            best_sse = (
                final_sse
            )

            best_params = insert_x1(
                result.x,
                fixed_x1
            )

    if best_params is None:

        print(
            f"WARNING: no valid optimization "
            f"found for x1={fixed_x1:.3f}"
        )

    return (
        best_sse,
        best_params,
        records
    )


# ============================================================
# 17. SAVE PARTIAL RESULTS
# ============================================================

def save_partial(
    result_dict,
    optimization_records
):

    rows = []

    for x1 in sorted(
        result_dict.keys()
    ):

        result = result_dict[
            x1
        ]

        row = {
            "x1": x1,
            "profile_sse": result[
                "sse"
            ]
        }

        params = result[
            "params"
        ]

        if params is not None:

            for name, value in zip(
                PARAM_NAMES,
                params
            ):

                row[name] = value

        rows.append(
            row
        )

    pd.DataFrame(
        rows
    ).to_csv(
        os.path.join(
            BASE_DIR,
            "x1_profile_sse_partial.csv"
        ),
        index=False
    )

    pd.DataFrame(
        optimization_records
    ).to_csv(
        os.path.join(
            BASE_DIR,
            "x1_profile_optimization_log_partial.csv"
        ),
        index=False
    )


# ============================================================
# 18. MAIN
# ============================================================

def main():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    print("=" * 72)
    print(
        "REFINED PROFILE-SSE ANALYSIS FOR x1"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Load polished unrestricted fit
    # --------------------------------------------------------

    polished_fit = load_polished_fit()

    polished_x1 = polished_fit[
        X1_INDEX
    ]

    polished_sse = calculate_sse(
        polished_fit
    )

    print(
        f"\nUnrestricted polished x1: "
        f"{polished_x1:.6f}"
    )

    print(
        f"Unrestricted polished SSE: "
        f"{polished_sse:.6f}"
    )

    # --------------------------------------------------------
    # Load LHS library
    # --------------------------------------------------------

    (
        candidate_params,
        candidate_sse
    ) = load_existing_library()

    print(
        f"\nValid LHS candidates: "
        f"{len(candidate_params)}"
    )

    print(
        f"Best raw LHS SSE: "
        f"{candidate_sse.min():.6f}"
    )

    # --------------------------------------------------------
    # Center grid at closest point to unrestricted x1
    # --------------------------------------------------------

    center_index = np.argmin(
        np.abs(
            X1_GRID
            - polished_x1
        )
    )

    center_x1 = X1_GRID[
        center_index
    ]

    print(
        f"\nStarting continuation near "
        f"x1={center_x1:.2f}"
    )

    results = {}

    optimization_records = []

    # ========================================================
    # A. CENTER
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        f"CENTER: x1="
        f"{center_x1:.2f}"
    )

    print(
        "=" * 72
    )

    (
        center_sse,
        center_params,
        center_records
    ) = optimize_fixed_x1(
        fixed_x1=center_x1,
        polished_fit=polished_fit,
        candidate_params=candidate_params,
        candidate_sse=candidate_sse,
        rng=rng,
        warm_start=polished_fit
    )

    results[
        center_x1
    ] = {
        "sse": center_sse,
        "params": center_params
    }

    optimization_records.extend(
        center_records
    )

    save_partial(
        results,
        optimization_records
    )

    # ========================================================
    # B. WALK DOWNWARD
    # ========================================================

    warm_params = center_params

    for idx in range(
        center_index - 1,
        -1,
        -1
    ):

        fixed_x1 = X1_GRID[
            idx
        ]

        print(
            "\n" + "=" * 72
        )

        print(
            f"DOWNWARD: x1="
            f"{fixed_x1:.2f}"
        )

        print(
            "=" * 72
        )

        (
            best_sse,
            best_params,
            records
        ) = optimize_fixed_x1(
            fixed_x1=fixed_x1,
            polished_fit=polished_fit,
            candidate_params=candidate_params,
            candidate_sse=candidate_sse,
            rng=rng,
            warm_start=warm_params
        )

        results[
            fixed_x1
        ] = {
            "sse": best_sse,
            "params": best_params
        }

        optimization_records.extend(
            records
        )

        if best_params is not None:
            warm_params = best_params

        save_partial(
            results,
            optimization_records
        )

    # ========================================================
    # C. WALK UPWARD
    # ========================================================

    warm_params = center_params

    for idx in range(
        center_index + 1,
        len(X1_GRID)
    ):

        fixed_x1 = X1_GRID[
            idx
        ]

        print(
            "\n" + "=" * 72
        )

        print(
            f"UPWARD: x1="
            f"{fixed_x1:.2f}"
        )

        print(
            "=" * 72
        )

        (
            best_sse,
            best_params,
            records
        ) = optimize_fixed_x1(
            fixed_x1=fixed_x1,
            polished_fit=polished_fit,
            candidate_params=candidate_params,
            candidate_sse=candidate_sse,
            rng=rng,
            warm_start=warm_params
        )

        results[
            fixed_x1
        ] = {
            "sse": best_sse,
            "params": best_params
        }

        optimization_records.extend(
            records
        )

        if best_params is not None:
            warm_params = best_params

        save_partial(
            results,
            optimization_records
        )

    # ========================================================
    # 19. FINAL TABLE
    # ========================================================

    rows = []

    for fixed_x1 in sorted(
        results.keys()
    ):

        result = results[
            fixed_x1
        ]

        row = {
            "x1": fixed_x1,
            "profile_sse": result[
                "sse"
            ]
        }

        params = result[
            "params"
        ]

        if params is not None:

            for name, value in zip(
                PARAM_NAMES,
                params
            ):

                row[name] = value

        else:

            for name in PARAM_NAMES:
                row[name] = np.nan

        rows.append(
            row
        )

    profile_df = pd.DataFrame(
        rows
    )

    finite = np.isfinite(
        profile_df[
            "profile_sse"
        ]
    )

    if not finite.any():
        raise RuntimeError(
            "No finite profile solutions found."
        )

    profile_min = profile_df.loc[
        finite,
        "profile_sse"
    ].min()

    profile_df[
        "delta_sse"
    ] = (
        profile_df[
            "profile_sse"
        ]
        - profile_min
    )

    best_idx = profile_df[
        "profile_sse"
    ].idxmin()

    best_x1 = profile_df.loc[
        best_idx,
        "x1"
    ]

    best_sse = profile_df.loc[
        best_idx,
        "profile_sse"
    ]

    # ========================================================
    # 20. SAVE FINAL FILES
    # ========================================================

    profile_df.to_csv(
        os.path.join(
            BASE_DIR,
            "x1_profile_sse_refined.csv"
        ),
        index=False
    )

    pd.DataFrame(
        optimization_records
    ).to_csv(
        os.path.join(
            BASE_DIR,
            "x1_profile_optimization_log_refined.csv"
        ),
        index=False
    )

    # ========================================================
    # 21. PLOT
    # ========================================================

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        profile_df[
            "x1"
        ],
        profile_df[
            "delta_sse"
        ],
        marker="o",
        markersize=3
    )

    plt.axvline(
        best_x1,
        linestyle="--",
        label=(
            rf"Profile minimum "
            rf"$x_1={best_x1:.2f}$"
        )
    )

    plt.axvline(
        polished_x1,
        linestyle=":",
        label=(
            rf"Unrestricted fit "
            rf"$x_1={polished_x1:.2f}$"
        )
    )

    plt.xlabel(
        r"Fixed decline-onset parameter "
        r"$x_1$ (weeks)"
    )

    plt.ylabel(
        r"$\Delta$SSE relative to "
        r"profile minimum"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            BASE_DIR,
            "x1_profile_sse_refined.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 22. SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "REFINED PROFILE COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nUnrestricted polished x1 "
        f"= {polished_x1:.4f}"
    )

    print(
        f"Unrestricted polished SSE "
        f"= {polished_sse:.4f}"
    )

    print(
        f"\nProfile minimum x1 "
        f"= {best_x1:.4f}"
    )

    print(
        f"Profile minimum SSE "
        f"= {best_sse:.4f}"
    )

    print(
        "\nBest 15 profile points:"
    )

    print(
        profile_df[
            [
                "x1",
                "profile_sse",
                "delta_sse"
            ]
        ]
        .sort_values(
            "profile_sse"
        )
        .head(15)
        .to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        "  x1_profile_sse_refined.csv"
    )

    print(
        "  x1_profile_optimization_log_refined.csv"
    )

    print(
        "  x1_profile_sse_refined.png"
    )


# ============================================================
# 23. RUN
# ============================================================

if __name__ == "__main__":
    main()