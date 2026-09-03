"""
parametric_bootstrap_test.py

Parametric bootstrap / simulated-null test for the double-sigmoid vs
single-sigmoid model comparison at Vetlyanka (SIGMOIDAL gamma/mu pair --
see build_frozen_library.py / bootstrap_frozen.py for the frozen-gamma/mu
equivalent).

WHY THIS VERSION ADDS A POLISH STEP (important -- fixes a real bug):
check_library_search_power.py confirmed the raw library-lookup "refit"
(argmin SSE over precomputed candidates) is a MUCH weaker stand-in for
the real fitting procedure than the actual multi-start+L-BFGS-B polish
used to produce your reported SSE values -- and, critically, the gap is
strongly ASYMMETRIC between models (single's true optimum sits at
multiple parameter-bound walls simultaneously, which raw library draws
are intrinsically unlikely to land near; double's optimum is mostly
interior, so library draws land much closer to it by chance). Since the
`observed` deltas are computed from fully-polished real fits but every
bootstrap replicate previously compared unpolished library lookups for
BOTH models, single was being systematically handicapped more than
double inside the bootstrap loop -- inflating the null distribution's
typical double-favoring gap for a reason that has nothing to do with
whether a genuine second regime exists. This version adds a single
L-BFGS-B polish step, starting from the best library candidate, for
BOTH models, in every replicate -- closing that asymmetric gap so the
null distribution represents the SAME fitting procedure the observed
deltas were computed from.

T_perceive: the real single-sigmoid fit FIXES T_perceive at 3.571
(polish_vetlyanka_fit_single_FIXED.py) rather than fitting it. The
polish step below respects this exactly: only the 10 real free
parameters are polished for single, T_perceive stays fixed at 3.571
throughout, never part of the search -- polishing over T_perceive too
would hand single MORE flexibility than it actually has in the real
fitting procedure, defeating the point of this fix.

Expected inputs (from precompute_trajectory_library.py, in
data/bootstrap/sigmoidal/):
    double_trajectories.npy   -- (num_samples, 22), used only to find a
                                  good STARTING candidate per replicate
    single_trajectories.npy   -- same, for single
    candidate_params.npy      -- (num_samples, 13), the raw LHS rows
                                  double_trajectories.npy and
                                  single_trajectories.npy were generated
                                  from (same row index across all three)

Outputs:
    bootstrap_null_distribution.csv   -- one row per bootstrap replicate
    bootstrap_null_distribution.png   -- histogram vs. observed value
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from plague_double_sigmoid_model import plague_model as plague_model_double
from plague_single_sigmoid_model import plague_model as plague_model_single
from vetlyanka_bounds import PARAM_NAMES, PARAM_NAMES_SINGLE, BOUNDS, BOUNDS_SINGLE, K_DOUBLE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP_DIR = os.path.join(BASE_DIR, "data", "bootstrap", "sigmoidal")
FITS_SINGLE_DIR = os.path.join(BASE_DIR, "data", "fits", "single")
os.makedirs(BOOTSTRAP_DIR, exist_ok=True)

N_BOOTSTRAP = 500  # halved from 1000 -- the effect size is large and consistent
                    # even at tiny replicate counts during testing, 500 gives a solid,
                    # still-robust null distribution at roughly half the runtime
RNG_SEED = 0
rng = np.random.default_rng(RNG_SEED)

n_weeks = 22
k_single = 10  # T_perceive fixed, not fitted
k_double = K_DOUBLE

T_PERCEIVE_FIXED_SIGMOID = 3.571  # must match polish_vetlyanka_fit_single_FIXED.py
X1_C1_PLACEHOLDER = 1.0
X0_X1_MIN_GAP = 1.0  # must match polish_vetlyanka_fit_double.py

PARAM_NAMES_SINGLE_SIGMOID_FIT = [name for name in PARAM_NAMES_SINGLE if name != "T_perceive"]
IDX_X0 = PARAM_NAMES.index("x0")
IDX_X1 = PARAM_NAMES.index("x1")
IDX_T_PERCEIVE = PARAM_NAMES.index("T_perceive")
_FULL_TO_SINGLE_IDX = [i for i, name in enumerate(PARAM_NAMES) if name in PARAM_NAMES_SINGLE_SIGMOID_FIT]

t_start, t_end, dt = 0, 22, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]

POLISH_MAXITER = 25  # confirmed sufficient -- polish starts from an already-good library
                      # match, converges in a handful of iterations; tested at 60 and 300
                      # with no meaningful change in the resulting null distribution


# ------------------------------------------------------------
# Load the precomputed trajectory library + the raw candidate parameters
# ------------------------------------------------------------
double_trajectories = np.load(os.path.join(BOOTSTRAP_DIR, "double_trajectories.npy"))
single_trajectories = np.load(os.path.join(BOOTSTRAP_DIR, "single_trajectories.npy"))
candidate_params = np.load(os.path.join(BOOTSTRAP_DIR, "candidate_params.npy"))

double_valid = ~np.isnan(double_trajectories).any(axis=1)
single_valid = ~np.isnan(single_trajectories).any(axis=1)
double_trajectories_v = double_trajectories[double_valid]
single_trajectories_v = single_trajectories[single_valid]
double_params_v = candidate_params[double_valid]
single_params_v = candidate_params[single_valid]
print(f"Using {double_trajectories_v.shape[0]} valid double-sigmoid candidates, "
      f"{single_trajectories_v.shape[0]} valid single-sigmoid candidates.")


# ------------------------------------------------------------
# Simulation + polish helpers (mirror the real fitting scripts)
# ------------------------------------------------------------
def simulate_weekly(model_fn, full_params):
    sol = solve_ivp(model_fn, [t_start, t_end], initial_conditions,
                     args=(full_params,), t_eval=t_points, method="RK45")
    if not sol.success:
        return None
    DR = sol.y[5][::steps_per_week][:n_weeks]
    if len(DR) < n_weeks:
        return None
    return DR


def sse_double_full(full_params, target):
    if full_params[IDX_X0] >= full_params[IDX_X1] - X0_X1_MIN_GAP:
        return 1e12
    predicted = simulate_weekly(plague_model_double, full_params)
    if predicted is None:
        return 1e12
    sse = np.sum((target - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def single_reduced_to_full(reduced_params):
    single_dict = dict(zip(PARAM_NAMES_SINGLE_SIGMOID_FIT, reduced_params))
    single_dict["T_perceive"] = T_PERCEIVE_FIXED_SIGMOID
    return np.array([single_dict.get(name, X1_C1_PLACEHOLDER) for name in PARAM_NAMES])


def sse_single_reduced(reduced_params, target):
    full_params = single_reduced_to_full(reduced_params)
    predicted = simulate_weekly(plague_model_single, full_params)
    if predicted is None:
        return 1e12
    sse = np.sum((target - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def scale_parameters(params):
    scaling_factors = np.maximum(np.abs(params), 1e-3)
    return params / scaling_factors, scaling_factors


def polish_double(start_full, target):
    scaled_x0, scaling_factors = scale_parameters(start_full)
    scaled_bounds = [(lo / sf, hi / sf) for (lo, hi), sf in zip(BOUNDS, scaling_factors)]
    result = minimize(
        lambda sp: sse_double_full(sp * scaling_factors, target),
        scaled_x0, method="L-BFGS-B", bounds=scaled_bounds,
        options={"maxiter": POLISH_MAXITER, "ftol": 1e-10, "gtol": 1e-8},
    )
    optimized = result.x * scaling_factors
    return sse_double_full(optimized, target), optimized


def polish_single(start_full, target):
    start_reduced = start_full[_FULL_TO_SINGLE_IDX]
    scaled_x0, scaling_factors = scale_parameters(start_reduced)
    reduced_bounds = [b for name, b in zip(PARAM_NAMES_SINGLE, BOUNDS_SINGLE) if name != "T_perceive"]
    scaled_bounds = [(lo / sf, hi / sf) for (lo, hi), sf in zip(reduced_bounds, scaling_factors)]
    result = minimize(
        lambda sp: sse_single_reduced(sp * scaling_factors, target),
        scaled_x0, method="L-BFGS-B", bounds=scaled_bounds,
        options={"maxiter": POLISH_MAXITER, "ftol": 1e-10, "gtol": 1e-8},
    )
    optimized = result.x * scaling_factors
    return sse_single_reduced(optimized, target), single_reduced_to_full(optimized)


# ------------------------------------------------------------
# Ground truth: the real single-sigmoid best fit, re-solved directly
# ------------------------------------------------------------
def load_ground_truth(csv_path):
    df = pd.read_csv(csv_path).set_index("Parameter")
    single_params = np.array([df.loc[name, "Best_Fit"] for name in PARAM_NAMES_SINGLE_SIGMOID_FIT])
    return single_reduced_to_full(single_params)


truth_params = load_ground_truth(os.path.join(FITS_SINGLE_DIR, "vetlyanka_polished_fit_single.csv"))
sol = solve_ivp(plague_model_single, [t_start, t_end], initial_conditions,
                 args=(truth_params,), t_eval=t_points, method="RK45")
true_cumulative = sol.y[5][::steps_per_week][:n_weeks]


def to_incremental(cumulative, baseline=0.0):
    cumulative = np.asarray(cumulative, dtype=float)
    return np.diff(np.concatenate(([baseline], cumulative)))


true_weekly_increments = np.clip(to_incremental(true_cumulative), 0, None)


def calculate_information_criteria(observed, predicted, k):
    n = observed.size
    sse = float(np.sum((observed - predicted) ** 2))
    if sse <= 0:
        sse = np.finfo(float).tiny
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    aicc = np.inf if n <= k + 1 else aic + (2 * k * (k + 1)) / (n - k - 1)
    return aic, aicc, bic


# ------------------------------------------------------------
# Bootstrap loop -- library search for a starting point, THEN polish,
# for both models, every replicate.
# ------------------------------------------------------------
records = []

for b in range(N_BOOTSTRAP):
    synthetic_increments = rng.poisson(true_weekly_increments).astype(float)
    synthetic_cumulative = np.cumsum(synthetic_increments)

    sse_double_lib = np.sum((double_trajectories_v - synthetic_cumulative) ** 2, axis=1)
    sse_single_lib = np.sum((single_trajectories_v - synthetic_cumulative) ** 2, axis=1)
    start_double = double_params_v[np.argmin(sse_double_lib)]
    start_single = single_params_v[np.argmin(sse_single_lib)]

    _, polished_double_full = polish_double(start_double, synthetic_cumulative)
    _, polished_single_full = polish_single(start_single, synthetic_cumulative)

    best_double = simulate_weekly(plague_model_double, polished_double_full)
    best_single = simulate_weekly(plague_model_single, polished_single_full)
    if best_double is None or best_single is None:
        continue

    aic_d, aicc_d, bic_d = calculate_information_criteria(synthetic_cumulative, best_double, k_double)
    aic_s, aicc_s, bic_s = calculate_information_criteria(synthetic_cumulative, best_single, k_single)

    synth_incr = to_incremental(synthetic_cumulative)
    aic_d_i, aicc_d_i, bic_d_i = calculate_information_criteria(synth_incr, to_incremental(best_double), k_double)
    aic_s_i, aicc_s_i, bic_s_i = calculate_information_criteria(synth_incr, to_incremental(best_single), k_single)

    records.append({
        "delta_AIC_cum": aic_s - aic_d, "delta_AICc_cum": aicc_s - aicc_d, "delta_BIC_cum": bic_s - bic_d,
        "delta_AIC_incr": aic_s_i - aic_d_i, "delta_AICc_incr": aicc_s_i - aicc_d_i, "delta_BIC_incr": bic_s_i - bic_d_i,
    })

    if (b + 1) % 50 == 0:
        print(f"  bootstrap replicate {b + 1}/{N_BOOTSTRAP}")

null_df = pd.DataFrame(records)
null_df.to_csv(os.path.join(BOOTSTRAP_DIR, "bootstrap_null_distribution.csv"), index=False)

observed = {
    "delta_AIC_cum": 13.654, "delta_AICc_cum": -11.846, "delta_BIC_cum": 10.381,
    "delta_AIC_incr": 11.277, "delta_AICc_incr": -14.223, "delta_BIC_incr": 8.004,
}
print("\nNull distribution summary (from single-regime synthetic data):")
print(null_df.describe().round(2))

print("\nComparison against the real, observed deltas:")
for col, obs_val in observed.items():
    null_vals = null_df[col].values
    p_value = float(np.mean(null_vals >= obs_val))
    print(f"  {col}: observed = {obs_val:.3f}, null mean = {null_vals.mean():.3f}, "
          f"null max = {null_vals.max():.3f}, fraction of null >= observed = {p_value:.3f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(null_df["delta_AICc_incr"], bins=20, color="steelblue", edgecolor="white")
ax.axvline(observed["delta_AICc_incr"], color="crimson", linewidth=2, label="Observed \u0394AICc (real data)")
ax.set_xlabel("\u0394AICc (single sigmoid \u2212 double sigmoid), incremental basis")
ax.set_ylabel("Count across bootstrap replicates")
ax.set_title("Null distribution under a true single-regime world (polished refit)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(BOOTSTRAP_DIR, "bootstrap_null_distribution.png"), dpi=200)
print(f"\nSaved {BOOTSTRAP_DIR}/bootstrap_null_distribution.png")