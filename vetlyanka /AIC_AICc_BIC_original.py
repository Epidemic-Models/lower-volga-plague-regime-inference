"""
AIC_AICc_BIC_original.py

Model comparison across THREE models for the Vetlyanka fit:
  - "Null"   -- single-sigmoid transmission, behavioral feedback DISABLED
               (sensitivity fixed at 0). 9 free parameters. See
               polish_vetlyanka_fit_null.py.
  - "Single" -- single-sigmoid transmission with behavioral feedback.
               10 free parameters -- T_perceive is FIXED at 3.571 (not
               searched -- see polish_vetlyanka_fit_single_FIXED.py's own
               docstring: widening its bound repeatedly revealed a joint
               escape route with dispose_rate, resolved by fixing
               T_perceive as a constant). It is NOT a row in
               vetlyanka_polished_fit_single.csv and must be supplied
               here as the same fixed constant, not read from the file.
  - "Double" -- double-sigmoid transmission with behavioral feedback.
               13 free parameters. See polish_vetlyanka_fit_double.py.

REQUIRED BEFORE RUNNING:
  - vetlyanka_polished_fit.csv        (from polish_vetlyanka_fit_double.py)
  - vetlyanka_polished_fit_single.csv (from polish_vetlyanka_fit_single_FIXED.py,
                                        10 rows -- no T_perceive row)
  - vetlyanka_polished_fit_null.csv   (from polish_vetlyanka_fit_null.py)
  - plague_double_sigmoid_model.py, plague_single_sigmoid_model.py,
    vetlyanka_bounds.py, all in the same directory.
"""
import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from plague_double_sigmoid_model import plague_model as plague_model_double
from plague_single_sigmoid_model import plague_model as plague_model_single
from vetlyanka_bounds import (
    PARAM_NAMES, PARAM_NAMES_SINGLE, PARAM_NAMES_NULL,
    K_DOUBLE, K_NULL,
)



# Ordinary single-sigmoid model: T_perceive is now FIXED, not fitted --
# 10 free parameters, not the imported K_SINGLE=11 (that constant is
# stale relative to polish_vetlyanka_fit_single_FIXED.py's current design).
K_SINGLE_SIGMOID = 10
T_PERCEIVE_FIXED_SIGMOID = 3.571  # must match polish_vetlyanka_fit_single_FIXED.py

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPARISONS_DIR = os.path.join(BASE_DIR, "data", "comparisons")
os.makedirs(COMPARISONS_DIR, exist_ok=True)

t_start, t_end, dt = 0, 22, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
n_weeks = 22
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]

X1_C1_PLACEHOLDER = 1.0        # inert for the single-sigmoid model (beta(t) ignores them)
T_PERCEIVE_PLACEHOLDER = 1.0   # irrelevant once sensitivity=0 (avoids div-by-zero in dP/dt)
SENSITIVITY_FIXED = 0.0        # the null model's defining constraint

# The 10 parameter names actually present in vetlyanka_polished_fit_single.csv
# now that T_perceive is fixed there too (PARAM_NAMES_SINGLE from
# vetlyanka_bounds.py still lists 11 names including T_perceive -- that
# describes the model's historical search space, not this fit's current CSV).
PARAM_NAMES_SINGLE_SIGMOID_FIT = [name for name in PARAM_NAMES_SINGLE if name != "T_perceive"]

observed_cumulative = np.array([
    3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19,
    27, 34, 90, 259, 313, 345, 364, 376,
    376, 376, 376
], dtype=float)


def load_best_fit(csv_path: str, param_order: list) -> np.ndarray:
    """Load Best_Fit values by parameter NAME (not row position)."""
    df = pd.read_csv(csv_path).set_index("Parameter")
    return np.array([df.loc[name, "Best_Fit"] for name in param_order])


def expand_single_to_full(single_params: np.ndarray) -> np.ndarray:
    """
    Expand the 10-parameter ordinary single-sigmoid fit to the full
    13-slot vector. T_perceive is FIXED (not read from the file, not
    part of single_params) -- supplied directly as T_PERCEIVE_FIXED_SIGMOID.
    """
    single_dict = dict(zip(PARAM_NAMES_SINGLE_SIGMOID_FIT, single_params))
    single_dict["T_perceive"] = T_PERCEIVE_FIXED_SIGMOID
    return np.array([
        single_dict.get(name, X1_C1_PLACEHOLDER) for name in PARAM_NAMES
    ])


def expand_null_to_full(null_params: np.ndarray) -> np.ndarray:
    null_dict = dict(zip(PARAM_NAMES_NULL, null_params))
    full = []
    for name in PARAM_NAMES:
        if name == 'sensitivity':
            full.append(SENSITIVITY_FIXED)
        elif name == 'T_perceive':
            full.append(T_PERCEIVE_PLACEHOLDER)
        elif name in null_dict:
            full.append(null_dict[name])
        else:  # x1, c1
            full.append(X1_C1_PLACEHOLDER)
    return np.array(full)


def simulate_weekly_cumulative(model_fn, full_params: np.ndarray) -> np.ndarray:
    sol = solve_ivp(
        model_fn, [t_start, t_end], initial_conditions,
        args=(full_params,), t_eval=t_points, method="RK45",
    )
    if not sol.success:
        raise RuntimeError(
            "ODE solve failed for the reported best-fit parameters "
            "-- check the CSV values are sane before trusting this."
        )
    DR = sol.y[5]
    weekly = DR[::steps_per_week][:n_weeks]
    if len(weekly) < n_weeks:
        raise RuntimeError(f"Only got {len(weekly)} weekly points, expected {n_weeks}.")
    return weekly


def calculate_information_criteria(observed: np.ndarray, predicted: np.ndarray, k: int) -> dict:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape:
        raise ValueError(
            f"Observed shape {observed.shape} does not match predicted shape {predicted.shape}."
        )

    n = observed.size
    residuals = observed - predicted
    sse = float(np.sum(residuals ** 2))
    rmse = float(np.sqrt(sse / n))
    if sse <= 0:
        sse = np.finfo(float).tiny

    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    aicc = np.inf if n <= k + 1 else aic + (2 * k * (k + 1)) / (n - k - 1)

    return {"n": n, "k": k, "SSE": sse, "RMSE": rmse, "AIC": float(aic), "AICc": float(aicc), "BIC": float(bic)}


def to_incremental(cumulative: np.ndarray, baseline: float = 0.0) -> np.ndarray:
    cumulative = np.asarray(cumulative, dtype=float)
    prepended = np.concatenate(([baseline], cumulative))
    return np.diff(prepended)


def print_pairwise_deltas(stats_a, label_a, stats_b, label_b, basis):
    d_aic = stats_a["AIC"] - stats_b["AIC"]
    d_aicc = stats_a["AICc"] - stats_b["AICc"]
    d_bic = stats_a["BIC"] - stats_b["BIC"]
    print(f"  {label_a} minus {label_b} ({basis}): "
          f"dAIC={d_aic:.3f}  dAICc={d_aicc:.3f}  dBIC={d_bic:.3f}  "
          f"(positive favours {label_b})")


double_best = load_best_fit(
    os.path.join(BASE_DIR, "data", "fits", "double", "vetlyanka_polished_fit.csv"),
    PARAM_NAMES,
)
single_best_full = expand_single_to_full(
    load_best_fit(
        os.path.join(BASE_DIR, "data", "fits", "single", "vetlyanka_polished_fit_single.csv"),
        PARAM_NAMES_SINGLE_SIGMOID_FIT,
    )
)
null_best_full = expand_null_to_full(
    load_best_fit(
        os.path.join(BASE_DIR, "data", "fits", "null", "vetlyanka_polished_fit_null.csv"),
        PARAM_NAMES_NULL,
    )
)

double_predicted = simulate_weekly_cumulative(plague_model_double, double_best)
single_predicted = simulate_weekly_cumulative(plague_model_single, single_best_full)
null_predicted = simulate_weekly_cumulative(plague_model_single, null_best_full)

null_stats = calculate_information_criteria(observed_cumulative, null_predicted, k=K_NULL)
single_stats = calculate_information_criteria(observed_cumulative, single_predicted, k=K_SINGLE_SIGMOID)
double_stats = calculate_information_criteria(observed_cumulative, double_predicted, k=K_DOUBLE)

results = pd.DataFrame(
    [null_stats, single_stats, double_stats],
    index=["Null (no feedback)", "Single-sigmoid (+feedback)", "Double-sigmoid"],
)
print("\nModel comparison (cumulative basis)")
print(results.round(3))
print()
print_pairwise_deltas(null_stats, "Null", single_stats, "Single", "cumulative")
print_pairwise_deltas(single_stats, "Single", double_stats, "Double", "cumulative")
print_pairwise_deltas(null_stats, "Null", double_stats, "Double", "cumulative")

results.to_csv("vetlyanka_model_comparison.csv")
print("\nSaved: vetlyanka_model_comparison.csv")

observed_incremental = to_incremental(observed_cumulative)
null_predicted_incremental = to_incremental(null_predicted)
single_predicted_incremental = to_incremental(single_predicted)
double_predicted_incremental = to_incremental(double_predicted)

null_stats_incr = calculate_information_criteria(
    observed_incremental, null_predicted_incremental, k=K_NULL
)
single_stats_incr = calculate_information_criteria(
    observed_incremental, single_predicted_incremental, k=K_SINGLE_SIGMOID
)
double_stats_incr = calculate_information_criteria(
    observed_incremental, double_predicted_incremental, k=K_DOUBLE
)

results_incr = pd.DataFrame(
    [null_stats_incr, single_stats_incr, double_stats_incr],
    index=["Null (no feedback, incremental)", "Single-sigmoid (+feedback, incremental)", "Double-sigmoid (incremental)"],
)
print("\nModel comparison (incremental basis)")
print(results_incr.round(3))
print()
print_pairwise_deltas(null_stats_incr, "Null", single_stats_incr, "Single", "incremental")
print_pairwise_deltas(single_stats_incr, "Single", double_stats_incr, "Double", "incremental")
print_pairwise_deltas(null_stats_incr, "Null", double_stats_incr, "Double", "incremental")

results_incr.to_csv("vetlyanka_model_comparison_incremental.csv")
print("\nSaved: vetlyanka_model_comparison_incremental.csv")

print("Cumulative SSEs:", null_stats["SSE"], single_stats["SSE"], double_stats["SSE"])
print("Incremental SSEs:", null_stats_incr["SSE"], single_stats_incr["SSE"], double_stats_incr["SSE"])