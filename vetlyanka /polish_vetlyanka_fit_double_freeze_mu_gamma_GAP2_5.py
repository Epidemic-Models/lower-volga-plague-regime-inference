"""
polish_vetlyanka_fit_double_freeze_mu_gamma_FIXED.py

Full fit of the standalone Vetlyanka model with constant recovery and
mortality rates.

The original double-sigmoid model has 13 parameters:

    b1, b2, x0, x1, c, c1,
    gamma1, gamma2, mu1, mu2,
    T_perceive, sensitivity, dispose_rate

This model replaces:

    gamma1, gamma2 -> gamma_const
    mu1, mu2       -> mu_const

using the SPAN of each pair's bounds (min of both lows, max of both
highs), not their average -- averaging produced a gamma_const range with
zero overlap with either original range, and shrank mu_const's ceiling
well below mu2's real, historically-grounded ceiling. See vetlyanka_bounds.py.

Therefore, the reduced model has 11 fitted parameters.

MULTI-START: polishes the TOP_K best LHS candidates independently (not
just the single overall best one) and keeps whichever L-BFGS-B run
converges to the lowest true SSE. A single-start polish can converge
prematurely from a bad starting point -- few iterations, few function
evaluations, landing on wall-pinned parameters far from the real optimum
-- without scipy raising any error; it reports "success" based on its
own local stopping criterion, not on having found the actual best
achievable fit. This was observed directly: a single-start run landed at
SSE=778 with only 17 iterations / 300 evaluations, while a 5-way
multi-start on the same data found SSE=383 from a start that was NOT the
single best raw LHS candidate. See vetlyanka_constant_mu_gamma_multistart_log.csv
after running for the full comparison across all K starts.

The existing plague_model is reused. Before each simulation, the reduced
parameter vector is expanded to the original 13-parameter format by setting:

    gamma1 = gamma2 = gamma_const
    mu1    = mu2    = mu_const

Outputs are written to:

    freeze_mu_gamma/

REQUIRED BEFORE RUNNING:
  Set NUM_SAMPLES below to match an existing parameter_samples_<NUM_SAMPLES>.csv
  (from parameter_sampling.py). This project used 400,000 for the frozen
  fits specifically, since the span-widened gamma_const/mu_const bounds
  need denser search coverage than the 200,000 used for the original
  sigmoidal-gamma/mu fits -- see vetlyanka_bounds.py and
  parameter_sampling.py for why.

Required files:
    plague_double_sigmoid_model.py
    vetlyanka_bounds.py
    parameter_samples_<NUM_SAMPLES>.csv
"""

import os
import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from plague_double_sigmoid_model import plague_model
from vetlyanka_bounds import PARAM_NAMES, BOUNDS


# ---------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "freeze_mu_gamma_gap2_5")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 400000  # must match an existing parameter_samples_<NUM_SAMPLES>.csv
SAMPLES_PATH = os.path.join(BASE_DIR, f"parameter_samples_{NUM_SAMPLES}.csv")
TOP_K = 10  # polish this many independent starting points, keep the best
X0_X1_MIN_GAP = 2.5  # require x1 - x0 >= this many weeks -- raised from 1.0 after
                      # the previous run landed at gap=1.029, too close to the
                      # constraint boundary to tell if it's a genuine answer or
                      # still being pulled toward the blocked degenerate region

FIT_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "vetlyanka_constant_mu_gamma_polished_fit.csv",
)

PREDICTIONS_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "vetlyanka_constant_mu_gamma_predictions.csv",
)

SUMMARY_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "vetlyanka_constant_mu_gamma_summary.txt",
)

MULTISTART_LOG_PATH = os.path.join(
    OUTPUT_DIR,
    "vetlyanka_constant_mu_gamma_multistart_log.csv",
)


# ---------------------------------------------------------------------
# Time settings and initial conditions
# ---------------------------------------------------------------------

t_start = 0.0
t_end = 22.0
dt = 0.01

t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
n_weeks = 22

# S0, I0, R0, DI0, DDI0, DR0, P0
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]


# ---------------------------------------------------------------------
# Verify the expected full parameter order
# ---------------------------------------------------------------------

EXPECTED_FULL_NAMES = [
    "b1",
    "b2",
    "x0",
    "x1",
    "c",
    "c1",
    "gamma1",
    "gamma2",
    "mu1",
    "mu2",
    "T_perceive",
    "sensitivity",
    "dispose_rate",
]


def normalize_parameter_name(name):
    """
    Normalize LaTeX-like or alternative parameter labels so that the script
    can verify the ordering without requiring identical typography.
    """
    return (
        str(name)
        .replace("$", "")
        .replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .replace("text", "")
        .replace("mathrm", "")
        .replace(" ", "")
        .lower()
    )


normalized_actual = [normalize_parameter_name(x) for x in PARAM_NAMES]
normalized_expected = [normalize_parameter_name(x) for x in EXPECTED_FULL_NAMES]

if normalized_actual != normalized_expected:
    raise ValueError(
        "Unexpected parameter ordering in vetlyanka_bounds.py.\n"
        f"Expected:\n{EXPECTED_FULL_NAMES}\n\n"
        f"Found:\n{list(PARAM_NAMES)}\n\n"
        "Update EXPECTED_FULL_NAMES or the index mapping before fitting."
    )

if len(BOUNDS) != 13:
    raise ValueError(
        f"Expected 13 full-model bounds, but found {len(BOUNDS)}."
    )


# ---------------------------------------------------------------------
# Reduced parameter structure
# ---------------------------------------------------------------------

REDUCED_PARAM_NAMES = [
    "b1",
    "b2",
    "x0",
    "x1",
    "c",
    "c1",
    "gamma_const",
    "mu_const",
    "T_perceive",
    "sensitivity",
    "dispose_rate",
]


def span_bound(bound_1, bound_2):
    """
    Union of two bounds -- the smallest range that fully contains both
    original, independently-defensible intervals. NOT their average:
    averaging endpoints can produce a range with zero overlap with either
    original bound (this happened for gamma_const under the earlier,
    buggy version of this function).
    """
    lo_1, hi_1 = bound_1
    lo_2, hi_2 = bound_2
    if None in (lo_1, hi_1, lo_2, hi_2):
        raise ValueError("Constant gamma and mu require finite original bounds.")
    return (min(lo_1, lo_2), max(hi_1, hi_2))


GAMMA_CONST_BOUND = span_bound(BOUNDS[6], BOUNDS[7])
MU_CONST_BOUND = span_bound(BOUNDS[8], BOUNDS[9])

REDUCED_BOUNDS = [
    BOUNDS[0],              # b1
    BOUNDS[1],              # b2
    BOUNDS[2],              # x0
    BOUNDS[3],              # x1
    BOUNDS[4],              # c
    BOUNDS[5],              # c1
    GAMMA_CONST_BOUND,      # gamma_const
    MU_CONST_BOUND,         # mu_const
    BOUNDS[10],             # T_perceive
    BOUNDS[11],             # sensitivity
    BOUNDS[12],             # dispose_rate
]

K_REDUCED = len(REDUCED_PARAM_NAMES)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def load_observed_data():
    """Real Vetlyanka weekly cumulative death data."""
    cumulativeweekly_all_regions = np.array([
        [
            3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19,
            27, 34, 90, 259, 313, 345, 364, 376,
            376, 376, 376,
        ]
    ])

    observed = cumulativeweekly_all_regions[0].astype(float)

    if observed.shape[0] != n_weeks:
        raise ValueError(
            f"Expected {n_weeks} weekly observations, "
            f"but found {observed.shape[0]}."
        )

    return observed


# ---------------------------------------------------------------------
# Parameter conversion
# ---------------------------------------------------------------------

def full_to_reduced(full_params):
    """
    Convert a 13-parameter full-model vector to the 11-parameter
    constant-gamma/constant-mu representation, for scanning the SHARED
    (double-sigmoid-shaped) LHS sample file as candidate starting points.

    NOTE: this collapse (simple averaging of gamma1/gamma2 and mu1/mu2)
    is only used here to convert an arbitrary EXISTING 13-slot sample row
    into a reduced-space STARTING GUESS -- it is not the same as, and
    does not need to match, the span-based REDUCED_BOUNDS above, which
    define where the optimizer is actually allowed to search.
    """
    full_params = np.asarray(full_params, dtype=float)

    if full_params.shape != (13,):
        raise ValueError(
            f"Expected full parameter vector of length 13, "
            f"got shape {full_params.shape}."
        )

    gamma_const = 0.5 * (full_params[6] + full_params[7])
    mu_const = 0.5 * (full_params[8] + full_params[9])

    return np.array([
        full_params[0],     # b1
        full_params[1],     # b2
        full_params[2],     # x0
        full_params[3],     # x1
        full_params[4],     # c
        full_params[5],     # c1
        gamma_const,
        mu_const,
        full_params[10],    # T_perceive
        full_params[11],    # sensitivity
        full_params[12],    # dispose_rate
    ], dtype=float)


def reduced_to_full(reduced_params):
    """
    Expand an 11-parameter constant-rate vector into the original
    13-parameter format expected by plague_model.
    """
    reduced_params = np.asarray(reduced_params, dtype=float)

    if reduced_params.shape != (K_REDUCED,):
        raise ValueError(
            f"Expected reduced vector of length {K_REDUCED}, "
            f"got shape {reduced_params.shape}."
        )

    (
        b1,
        b2,
        x0,
        x1,
        c,
        c1,
        gamma_const,
        mu_const,
        T_perceive,
        sensitivity,
        dispose_rate,
    ) = reduced_params

    return np.array([
        b1,
        b2,
        x0,
        x1,
        c,
        c1,
        gamma_const,    # gamma1
        gamma_const,    # gamma2
        mu_const,       # mu1
        mu_const,       # mu2
        T_perceive,
        sensitivity,
        dispose_rate,
    ], dtype=float)


# ---------------------------------------------------------------------
# Simulation and objective
# ---------------------------------------------------------------------

def simulate_full_trajectory(reduced_params):
    """
    Solve the ODE system and return the complete solve_ivp solution.
    """
    full_params = reduced_to_full(reduced_params)

    try:
        sol = solve_ivp(
            plague_model,
            [t_start, t_end],
            initial_conditions,
            args=(full_params,),
            t_eval=t_points,
            method="RK45",
        )
    except (ValueError, FloatingPointError, OverflowError):
        return None

    if not sol.success:
        return None

    if not np.all(np.isfinite(sol.y)):
        return None

    return sol


def simulate_cumulative(reduced_params):
    """
    Return weekly cumulative deaths, DR, from the reduced model.
    """
    sol = simulate_full_trajectory(reduced_params)

    if sol is None:
        return None

    predicted = sol.y[5][::steps_per_week][:n_weeks]

    if predicted.shape[0] != n_weeks:
        return None

    if not np.all(np.isfinite(predicted)):
        return None

    return predicted


def scale_parameters(params):
    scaling_factors = np.maximum(np.abs(params), 1e-3)
    scaled_params = params / scaling_factors
    return scaled_params, scaling_factors


def sse_unscaled(reduced_params, observed):
    """
    Raw SSE evaluated using the unscaled reduced parameter vector.
    """
    x0, x1 = reduced_params[2], reduced_params[3]
    if x0 >= x1 - X0_X1_MIN_GAP:  # require the rise-midpoint stay meaningfully before
                                  # the decline-onset -- otherwise the two sigmoids
                                  # overlap into a degenerate, non-physical shape
                                  # (see vetlyanka_bounds.py's x0/x1 discussion)
        return 1e12

    predicted = simulate_cumulative(reduced_params)

    if predicted is None:
        return 1e12

    residuals = observed - predicted
    sse = np.sum(residuals ** 2)

    if not np.isfinite(sse):
        return 1e12

    return float(sse)

def sse_objective(scaled_params, scaling_factors, observed):
    """
    Objective used by the optimizer.
    """
    reduced_params = scaled_params * scaling_factors
    return sse_unscaled(reduced_params, observed)


# ---------------------------------------------------------------------
# LHS candidate scan
# ---------------------------------------------------------------------

def load_full_lhs_samples(samples_path):
    """
    Load the shared 13-parameter LHS sample file.
    """
    if not os.path.exists(samples_path):
        raise FileNotFoundError(
            f"Could not find LHS sample file:\n{samples_path}\n"
            f"Run parameter_sampling.py with NUM_SAMPLES={NUM_SAMPLES} first."
        )

    samples = np.loadtxt(
        samples_path,
        delimiter=",",
        skiprows=1,
    )

    if samples.ndim == 1:
        samples = samples.reshape(1, -1)

    if samples.shape[1] != 13:
        raise ValueError(
            f"Expected 13 columns in {samples_path}, "
            f"but found {samples.shape[1]}."
        )

    return samples


def find_top_k_lhs_candidates(samples_path, observed, k=TOP_K):
    """
    Convert every full-model LHS row to the reduced constant-rate model,
    evaluate its SSE, and return the k best reduced candidates (not just
    the single best) -- see the multi-start note in the module docstring
    for why a single start is not reliable enough here.
    """
    full_samples = load_full_lhs_samples(samples_path)

    print(
        f"Scanning {full_samples.shape[0]} existing LHS candidates "
        "after replacing gamma and mu by their means..."
    )

    scored = []
    for i, full_row in enumerate(full_samples):
        reduced_row = full_to_reduced(full_row)
        sse = sse_unscaled(reduced_row, observed)
        scored.append((sse, reduced_row))

        if (i + 1) % 20000 == 0:
            best_so_far = min(s for s, _ in scored)
            print(
                f"  scanned {i + 1}/{full_samples.shape[0]} candidates "
                f"(best SSE so far: {best_so_far:.6f})"
            )

    scored.sort(key=lambda pair: pair[0])
    top_k = scored[:k]

    print(f"\nTop {k} LHS candidate SSEs: {[f'{s:.6f}' for s, _ in top_k]}")

    return top_k  # list of (sse, reduced_row) tuples


# ---------------------------------------------------------------------
# Diagnostics and output
# ---------------------------------------------------------------------

def calculate_aic(sse, n, k):
    """
    Gaussian least-squares AIC, excluding additive constants shared by models.

        AIC = n * log(SSE / n) + 2k

    This is suitable when the same observations and raw SSE definition are
    used for both model variants.
    """
    if sse <= 0 or not np.isfinite(sse):
        return np.nan

    return float(n * np.log(sse / n) + 2 * k)


def calculate_aicc(sse, n, k):
    """
    Small-sample corrected AIC.
    """
    aic = calculate_aic(sse, n, k)

    denominator = n - k - 1

    if not np.isfinite(aic) or denominator <= 0:
        return np.nan

    correction = (2 * k * (k + 1)) / denominator
    return float(aic + correction)


def approximate_confidence_intervals(
    result,
    optimized_params,
    scaling_factors,
):
    """
    Transform the L-BFGS-B inverse-Hessian approximation from scaled
    coordinates to the original parameter coordinates.

    These intervals are approximate and should be interpreted cautiously,
    particularly for parameters close to a bound -- prefer the top-1%-
    percentile-band method (frozen_confidence_intervals.py) for anything
    reported in the paper.
    """
    n_params = len(optimized_params)

    try:
        if isinstance(result.hess_inv, np.ndarray):
            hess_scaled = result.hess_inv
        else:
            hess_scaled = result.hess_inv.todense()

        hess_scaled = np.asarray(hess_scaled, dtype=float)

        if hess_scaled.shape != (n_params, n_params):
            raise ValueError(
                f"Unexpected Hessian shape: {hess_scaled.shape}"
            )

        scale_outer = np.outer(
            scaling_factors,
            scaling_factors,
        )

        covariance_approx = hess_scaled * scale_outer

        diagonal = np.diag(covariance_approx)
        diagonal = np.where(diagonal >= 0, diagonal, np.nan)

        standard_errors = np.sqrt(diagonal)
        lower = optimized_params - 1.96 * standard_errors
        upper = optimized_params + 1.96 * standard_errors

        # Respect the fitted parameter bounds in the displayed intervals.
        for i, (lo, hi) in enumerate(REDUCED_BOUNDS):
            if lo is not None and np.isfinite(lower[i]):
                lower[i] = max(lower[i], lo)

            if hi is not None and np.isfinite(upper[i]):
                upper[i] = min(upper[i], hi)

        return standard_errors, lower, upper

    except (AttributeError, ValueError, TypeError) as exc:
        print(
            "\nHessian inverse was not usable: "
            f"{exc}\n"
            "Bootstrap or profile-likelihood intervals would be more reliable."
        )

        nan_array = np.full(n_params, np.nan)
        return nan_array, nan_array.copy(), nan_array.copy()


def identify_near_bound_parameters(params, threshold=0.05):
    """
    Identify parameters within threshold * 100 percent of either bound.
    """
    flags = []

    for name, (lo, hi), value in zip(
        REDUCED_PARAM_NAMES,
        REDUCED_BOUNDS,
        params,
    ):
        if lo is None or hi is None:
            continue

        span = hi - lo

        if span <= 0:
            continue

        relative_position = (value - lo) / span

        if (
            relative_position < threshold
            or relative_position > (1.0 - threshold)
        ):
            flags.append({
                "parameter": name,
                "value": value,
                "relative_position": relative_position,
            })

    return flags


def save_predictions(observed, predicted):
    weeks = np.arange(n_weeks)

    prediction_df = pd.DataFrame({
        "Week": weeks,
        "Observed_cumulative_deaths": observed,
        "Predicted_cumulative_deaths": predicted,
        "Residual": observed - predicted,
    })

    prediction_df.to_csv(
        PREDICTIONS_OUTPUT_PATH,
        index=False,
    )


def save_summary(
    result,
    lhs_sse,
    polished_sse,
    aic,
    aicc,
    near_bound_flags,
):
    lines = [
        "Vetlyanka constant-gamma/constant-mu full fit (multi-start)",
        "=" * 50,
        "",
        f"Number of observations: {n_weeks}",
        f"Number of fitted parameters: {K_REDUCED}",
        f"Best LHS starting SSE (of top {TOP_K} polished): {lhs_sse:.12g}",
        f"Polished SSE: {polished_sse:.12g}",
        f"AIC: {aic:.12g}",
        f"AICc: {aicc:.12g}",
        f"Optimizer success: {result.success}",
        f"Optimizer status: {result.status}",
        f"Optimizer message: {result.message}",
        f"Optimizer iterations: {getattr(result, 'nit', 'not available')}",
        f"Objective evaluations: {getattr(result, 'nfev', 'not available')}",
        "",
        "Constant-rate constraints:",
        "gamma1 = gamma2 = gamma_const",
        "mu1 = mu2 = mu_const",
        "",
        f"See {os.path.basename(MULTISTART_LOG_PATH)} for all {TOP_K} starts compared.",
        "",
    ]

    if near_bound_flags:
        lines.append("Parameters within 5% of a bound:")

        for item in near_bound_flags:
            percentage = 100.0 * item["relative_position"]

            lines.append(
                f"  {item['parameter']}: "
                f"value={item['value']:.12g}, "
                f"position={percentage:.2f}%"
            )
    else:
        lines.append("No parameters were within 5% of a bound.")

    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


# ---------------------------------------------------------------------
# Main fitting procedure
# ---------------------------------------------------------------------

def main():
    observed = load_observed_data()

    print("\nConstant-gamma/constant-mu Vetlyanka fit (multi-start)")
    print("=" * 48)

    print("\nReduced parameter names:")
    for name, bound in zip(REDUCED_PARAM_NAMES, REDUCED_BOUNDS):
        print(f"  {name:15s} {bound}")

    top_candidates = find_top_k_lhs_candidates(SAMPLES_PATH, observed)

    best_result = None
    best_polished_sse = np.inf
    best_scaling_factors = None
    multistart_log = []

    for rank, (lhs_sse, start_row) in enumerate(top_candidates):
        scaled_x0, scaling_factors = scale_parameters(start_row)
        scaled_bounds = [
            (lo / sf if lo is not None else None, hi / sf if hi is not None else None)
            for (lo, hi), sf in zip(REDUCED_BOUNDS, scaling_factors)
        ]

        print(f"\nPolishing start #{rank + 1}/{len(top_candidates)} (LHS SSE={lhs_sse:.3f})...")
        result = minimize(
            sse_objective, scaled_x0,
            args=(scaling_factors, observed),
            method="L-BFGS-B",
            bounds=scaled_bounds,
            options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 100},
        )

        optimized_reduced_params = result.x * scaling_factors
        polished_sse = sse_unscaled(optimized_reduced_params, observed)

        print(f"  -> polished SSE={polished_sse:.6f}  (iters={result.nit}, evals={result.nfev})")
        multistart_log.append({
            "rank": rank + 1,
            "lhs_sse": lhs_sse,
            "polished_sse": polished_sse,
            "iterations": result.nit,
            "evaluations": result.nfev,
        })

        if polished_sse < best_polished_sse:
            best_polished_sse = polished_sse
            best_result = result
            best_scaling_factors = scaling_factors

    pd.DataFrame(multistart_log).to_csv(MULTISTART_LOG_PATH, index=False)
    print(f"\nSaved multi-start log: {MULTISTART_LOG_PATH}")

    best_rank = min(range(len(multistart_log)), key=lambda i: multistart_log[i]["polished_sse"]) + 1
    print(f"\nBest overall: rank #{best_rank} start, polished SSE = {best_polished_sse:.6f}")
    if best_rank != 1:
        print("NOTE: the single-best-LHS-candidate start was NOT the best after polishing --")
        print("this confirms the space has multiple local optima and single-start polishing")
        print("is not reliable here.")

    result = best_result
    scaling_factors = best_scaling_factors
    optimized_reduced_params = result.x * scaling_factors
    polished_sse = sse_unscaled(optimized_reduced_params, observed)
    lhs_sse = min(s for s, _ in top_candidates)

    predicted = simulate_cumulative(optimized_reduced_params)

    if predicted is None:
        raise RuntimeError(
            "The polished parameter vector failed during final simulation."
        )

    standard_errors, ci_lower, ci_upper = (
        approximate_confidence_intervals(
            result,
            optimized_reduced_params,
            scaling_factors,
        )
    )

    aic = calculate_aic(
        polished_sse,
        n=n_weeks,
        k=K_REDUCED,
    )

    aicc = calculate_aicc(
        polished_sse,
        n=n_weeks,
        k=K_REDUCED,
    )

    fit_df = pd.DataFrame({
        "Parameter": REDUCED_PARAM_NAMES,
        "Bound_lo": [bound[0] for bound in REDUCED_BOUNDS],
        "Bound_hi": [bound[1] for bound in REDUCED_BOUNDS],
        "Best_Fit": optimized_reduced_params,
        "Std_Error_approx": standard_errors,
        "CI_lower_approx": ci_lower,
        "CI_upper_approx": ci_upper,
    })

    fit_df.to_csv(
        FIT_OUTPUT_PATH,
        index=False,
    )

    save_predictions(
        observed,
        predicted,
    )

    near_bound_flags = identify_near_bound_parameters(
        optimized_reduced_params,
        threshold=0.05,
    )

    save_summary(
        result=result,
        lhs_sse=lhs_sse,
        polished_sse=polished_sse,
        aic=aic,
        aicc=aicc,
        near_bound_flags=near_bound_flags,
    )

    print("\nOptimization finished")
    print("-" * 48)
    print(f"Optimizer success: {result.success}")
    print(f"Optimizer message: {result.message}")
    print(f"Best LHS starting SSE (of top {TOP_K} polished): {lhs_sse:.6f}")
    print(f"Polished SSE: {polished_sse:.6f}")
    print(f"AIC: {aic:.6f}")
    print(f"AICc: {aicc:.6f}")

    print("\nBest-fit reduced parameters:")
    print(fit_df.to_string(index=False))

    expanded_full_params = reduced_to_full(
        optimized_reduced_params
    )

    print("\nExpanded 13-parameter vector passed to plague_model:")
    for name, value in zip(
        EXPECTED_FULL_NAMES,
        expanded_full_params,
    ):
        print(f"  {name:15s} = {value:.12g}")

    if near_bound_flags:
        names = [
            item["parameter"]
            for item in near_bound_flags
        ]

        print(
            "\nWARNING: parameters within 5% of a bound: "
            f"{names}"
        )
    else:
        print("\nNo fitted parameters are within 5% of a bound.")

    print("\nSaved outputs:")
    print(f"  {FIT_OUTPUT_PATH}")
    print(f"  {PREDICTIONS_OUTPUT_PATH}")
    print(f"  {SUMMARY_OUTPUT_PATH}")
    print(f"  {MULTISTART_LOG_PATH}")


if __name__ == "__main__":
    main()