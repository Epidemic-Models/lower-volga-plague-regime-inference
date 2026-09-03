"""
recovery_simulation_study.py

Parameter-recovery / calibration study addressing Reviewer #3's request
("fitting the models on simulated data would clearly strengthen the
manuscript... assessing the model's estimates using simulated data")
and Reviewer #1's concern ("whether the better fits simply emerge
because there are additional parameters").

HOW THIS DIFFERS FROM THE EXISTING PARAMETRIC BOOTSTRAP
(parametric_bootstrap_test.py / bootstrap_frozen.py):

  Bootstrap (already done):
    - Ground truth = SINGLE-sigmoid's real fit (the null: one regime).
    - Tests: "if single-regime were secretly true, how often would
      double still look this much better by chance alone?"
    - A SPECIFICITY / false-positive-control check.
    - Refitting is a fast nearest-neighbour lookup against a
      precomputed trajectory library, not the real fitting pipeline.

  This script (new):
    - Ground truth = DOUBLE-sigmoid's real fit (the alternative: two
      regimes genuinely exist).
    - Tests: "if two regimes genuinely exist, can our ACTUAL fitting
      procedure (full LHS + multi-start + x0/x1 constraint) recover
      the true parameters, and does single-sigmoid fail in the SAME
      diagnostic way (persistent tail, no real zero) it does on the
      real data?"
    - A SENSITIVITY / recoverability / calibration check -- the
      complementary analysis, not a repeat of the bootstrap.
    - Refitting uses the REAL fitting pipeline, exactly as used on the
      actual data, not a library-lookup shortcut.

Together the two checks answer different halves of "is this result
real": the bootstrap rules out chance under the null; this script
confirms the fitting procedure can actually detect and recover the
alternative when it's genuinely present, and that single-sigmoid's
real-data failure signature isn't an artifact of that particular
dataset.

For each of NUM_REPLICATES synthetic datasets (generated from the real,
final double-sigmoid fit with Poisson noise), this script:
  1. Refits DOUBLE-sigmoid to the synthetic data (multi-start, same
     x0/x1 constraint as the real fit) and compares recovered
     parameters to the true generating values.
  2. Refits SINGLE-sigmoid to the SAME synthetic data (multi-start).
  3. Checks the tail-residual signature for both refits (weeks 19-21)
     -- does double reproduce a real halt, does single show a
     persistent trickle, the same qualitative pattern found on the
     real data?
  4. Computes AIC/AICc/BIC for both refits on this synthetic dataset.

NUM_LHS_SAMPLES/TOP_K below are reduced from the real fit's 200,000/10
-- this script runs a full fit FOUR times per replicate (double LHS
scan, double polish, single LHS scan, single polish), so the per-run
budget is trimmed to keep total runtime reasonable. Scale up for a
final paper-quality version if compute allows.

Outputs:
    recovery_study_results.csv -- one row per replicate, with true vs
        recovered double-sigmoid parameters, both models' SSE/AIC, and
        each model's tail-residual behaviour.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from plague_double_sigmoid_model import plague_model as plague_model_double
from plague_single_sigmoid_model import plague_model as plague_model_single
from vetlyanka_bounds import (
    PARAM_NAMES, BOUNDS, PARAM_NAMES_SINGLE, BOUNDS_SINGLE, K_DOUBLE, K_SINGLE,
)

NUM_REPLICATES = 5
NUM_LHS_SAMPLES = 200000  # reduced -- see module docstring
TOP_K = 5
RNG_SEED = 0
X0_X1_MIN_GAP = 1.0  # must match polish_vetlyanka_fit_double.py
X1_C1_PLACEHOLDER = 1.0

t_start, t_end, dt = 0, 22, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
n_weeks = 22
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]

IDX_X0 = PARAM_NAMES.index("x0")
IDX_X1 = PARAM_NAMES.index("x1")

rng = np.random.default_rng(RNG_SEED)


def load_true_double_params(csv_path="vetlyanka_polished_fit.csv"):
    df = pd.read_csv(csv_path).set_index("Parameter")
    return np.array([df.loc[name, "Best_Fit"] for name in PARAM_NAMES], dtype=float)


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


def to_incremental(cumulative, baseline=0.0):
    cumulative = np.asarray(cumulative, dtype=float)
    prepended = np.concatenate(([baseline], cumulative))
    return np.diff(prepended)


def calculate_information_criteria(observed, predicted, k):
    n = observed.size
    residuals = observed - predicted
    sse = float(np.sum(residuals ** 2))
    if sse <= 0:
        sse = np.finfo(float).tiny
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    aicc = np.inf if n <= k + 1 else aic + (2 * k * (k + 1)) / (n - k - 1)
    return aic, aicc, bic, sse


# ------------------------------------------------------------
# Double-sigmoid fitting (multi-start, x0/x1 constraint) -- mirrors
# polish_vetlyanka_fit_double.py exactly.
# ------------------------------------------------------------
def sse_unscaled_double(params, observed):
    if params[IDX_X0] >= params[IDX_X1] - X0_X1_MIN_GAP:
        return 1e12
    predicted = simulate_weekly(plague_model_double, params)
    if predicted is None:
        return 1e12
    sse = np.sum((observed - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def sse_objective_double(scaled_params, scaling_factors, observed):
    return sse_unscaled_double(scaled_params * scaling_factors, observed)


def scale_parameters(params):
    scaling_factors = np.maximum(np.abs(params), 1e-3)
    return params / scaling_factors, scaling_factors


def fit_double(observed):
    lhs_samples = rng.uniform(
        low=[b[0] for b in BOUNDS], high=[b[1] for b in BOUNDS],
        size=(NUM_LHS_SAMPLES, K_DOUBLE),
    )
    scored = [(sse_unscaled_double(row, observed), row) for row in lhs_samples]
    scored.sort(key=lambda pair: pair[0])
    top_k = scored[:TOP_K]

    best_sse, best_params = np.inf, None
    for lhs_sse, start_row in top_k:
        scaled_x0, scaling_factors = scale_parameters(start_row)
        scaled_bounds = [(lo / sf, hi / sf) for (lo, hi), sf in zip(BOUNDS, scaling_factors)]
        result = minimize(sse_objective_double, scaled_x0, args=(scaling_factors, observed),
                           method="L-BFGS-B", bounds=scaled_bounds,
                           options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 100})
        optimized = result.x * scaling_factors
        sse = sse_unscaled_double(optimized, observed)
        if sse < best_sse:
            best_sse, best_params = sse, optimized
    return best_params, best_sse


# ------------------------------------------------------------
# Single-sigmoid fitting (multi-start) -- mirrors
# polish_vetlyanka_fit_single.py exactly.
# ------------------------------------------------------------
def expand_single_to_full(single_params):
    single_dict = dict(zip(PARAM_NAMES_SINGLE, single_params))
    return np.array([single_dict.get(name, X1_C1_PLACEHOLDER) for name in PARAM_NAMES], dtype=float)


def sse_unscaled_single(single_params, observed):
    full_params = expand_single_to_full(single_params)
    predicted = simulate_weekly(plague_model_single, full_params)
    if predicted is None:
        return 1e12
    sse = np.sum((observed - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def sse_objective_single(scaled_params, scaling_factors, observed):
    return sse_unscaled_single(scaled_params * scaling_factors, observed)


def fit_single(observed):
    lhs_samples = rng.uniform(
        low=[b[0] for b in BOUNDS_SINGLE], high=[b[1] for b in BOUNDS_SINGLE],
        size=(NUM_LHS_SAMPLES, K_SINGLE),
    )
    scored = [(sse_unscaled_single(row, observed), row) for row in lhs_samples]
    scored.sort(key=lambda pair: pair[0])
    top_k = scored[:TOP_K]

    best_sse, best_params = np.inf, None
    for lhs_sse, start_row in top_k:
        scaled_x0, scaling_factors = scale_parameters(start_row)
        scaled_bounds = [(lo / sf, hi / sf) for (lo, hi), sf in zip(BOUNDS_SINGLE, scaling_factors)]
        result = minimize(sse_objective_single, scaled_x0, args=(scaling_factors, observed),
                           method="L-BFGS-B", bounds=scaled_bounds,
                           options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 100})
        optimized = result.x * scaling_factors
        sse = sse_unscaled_single(optimized, observed)
        if sse < best_sse:
            best_sse, best_params = sse, optimized
    return best_params, best_sse


# ------------------------------------------------------------
# Main recovery study
# ------------------------------------------------------------
def main():
    true_double_params = load_true_double_params()
    sol = solve_ivp(plague_model_double, [t_start, t_end], initial_conditions,
                     args=(true_double_params,), t_eval=t_points, method="RK45")
    true_cumulative = sol.y[5][::steps_per_week][:n_weeks]
    true_weekly_increments = np.clip(to_incremental(true_cumulative), 0, None)

    print(f"True double-sigmoid tail (weeks 19-21): {np.round(true_weekly_increments[19:22], 2)}")
    print(f"Running {NUM_REPLICATES} synthetic replicates...\n")

    results = []
    for rep in range(NUM_REPLICATES):
        print(f"{'='*60}\nReplicate {rep + 1}/{NUM_REPLICATES}\n{'='*60}")
        synthetic_increments = rng.poisson(true_weekly_increments).astype(float)
        synthetic_cumulative = np.cumsum(synthetic_increments)

        print("Fitting double-sigmoid to synthetic data...")
        double_recovered, double_sse = fit_double(synthetic_cumulative)
        double_predicted = simulate_weekly(plague_model_double, double_recovered)
        double_tail = to_incremental(double_predicted)[19:22]

        print("Fitting single-sigmoid to synthetic data...")
        single_recovered, single_sse = fit_single(synthetic_cumulative)
        single_full = expand_single_to_full(single_recovered)
        single_predicted = simulate_weekly(plague_model_single, single_full)
        single_tail = to_incremental(single_predicted)[19:22]

        aic_d, aicc_d, bic_d, _ = calculate_information_criteria(synthetic_cumulative, double_predicted, K_DOUBLE)
        aic_s, aicc_s, bic_s, _ = calculate_information_criteria(synthetic_cumulative, single_predicted, K_SINGLE)

        synthetic_tail = synthetic_increments[19:22]
        print(f"  Synthetic data tail:  {np.round(synthetic_tail, 2)}")
        print(f"  Double refit tail:    {np.round(double_tail, 2)}  (SSE={double_sse:.2f})")
        print(f"  Single refit tail:    {np.round(single_tail, 2)}  (SSE={single_sse:.2f})")
        print(f"  dAIC (single-double): {aic_s - aic_d:.3f}   dBIC: {bic_s - bic_d:.3f}")

        row = {
            "replicate": rep + 1,
            "double_sse": double_sse, "single_sse": single_sse,
            "delta_AIC": aic_s - aic_d, "delta_AICc": aicc_s - aicc_d, "delta_BIC": bic_s - bic_d,
            "double_tail_wk19": double_tail[0], "double_tail_wk20": double_tail[1], "double_tail_wk21": double_tail[2],
            "single_tail_wk19": single_tail[0], "single_tail_wk20": single_tail[1], "single_tail_wk21": single_tail[2],
        }
        for name, true_val, recovered_val in zip(PARAM_NAMES, true_double_params, double_recovered):
            row[f"true_{name}"] = true_val
            row[f"recovered_{name}"] = recovered_val
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv("recovery_study_results.csv", index=False)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Mean double tail (should be near 0): "
          f"{results_df[['double_tail_wk19','double_tail_wk20','double_tail_wk21']].mean().mean():.3f}")
    print(f"Mean single tail (should be persistently positive): "
          f"{results_df[['single_tail_wk19','single_tail_wk20','single_tail_wk21']].mean().mean():.3f}")
    print(f"delta_AIC (single-double) across replicates: "
          f"mean={results_df['delta_AIC'].mean():.2f}, min={results_df['delta_AIC'].min():.2f}")
    print("\nSaved: recovery_study_results.csv")


if __name__ == "__main__":
    main()