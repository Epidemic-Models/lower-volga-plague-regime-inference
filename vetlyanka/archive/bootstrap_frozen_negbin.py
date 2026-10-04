"""
bootstrap_frozen_negbin.py

Negative-binomial sensitivity version of bootstrap_frozen.py for the
constant-gamma/mu ("frozen") double-sigmoid versus single-sigmoid
Vetlyanka models.

The original bootstrap generates synthetic weekly deaths from the fitted
single-regime null model using Poisson noise. This script repeats the same
parametric-bootstrap procedure while allowing extra-Poisson variation
through an NB2 negative-binomial observation process,

    E[Y_t]   = mu_t
    Var[Y_t] = mu_t + mu_t^2 / phi,

where mu_t is the fitted single-regime expected weekly mortality and phi
is an overdispersion parameter estimated from the observed weekly deaths
relative to the fitted single-regime trajectory. As phi -> infinity, the
negative-binomial model approaches the Poisson model.

The purpose is a sensitivity analysis: the deterministic epidemic models,
parameter bounds, trajectory libraries, starting-point search, L-BFGS-B
polishing, and information-criterion calculations are unchanged. Only the
stochastic mechanism used to generate bootstrap mortality counts differs.

T_perceive: the single-frozen model fixes T_perceive at 3.409, matching
polish_vetlyanka_fit_single_freeze_mu_gamma_FIXED.py. Only its eight true
free parameters are polished. The double-frozen model retains freely fitted
T_perceive and has 11 free parameters.

The existing deterministic trajectory libraries are reused; they do not
need to be regenerated because the change from Poisson to negative-binomial
noise affects only synthetic observation generation, not deterministic
model trajectories.

Outputs:
    bootstrap_null_distribution_frozen_negbin.csv
    bootstrap_null_distribution_frozen_negbin.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from plague_double_sigmoid_model import plague_model as plague_model_double
from plague_single_sigmoid_model import plague_model as plague_model_single
from vetlyanka_bounds import PARAM_NAMES, PARAM_NAMES_SINGLE, BOUNDS, BOUNDS_SINGLE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP_DIR = os.path.join(BASE_DIR, "data", "bootstrap", "frozen")
FITS_DOUBLE_FROZEN_DIR = os.path.join(BASE_DIR, "data", "fits", "double_frozen")
FITS_SINGLE_FROZEN_DIR = os.path.join(BASE_DIR, "data", "fits", "single_frozen")
os.makedirs(BOOTSTRAP_DIR, exist_ok=True)

N_BOOTSTRAP = 500
RNG_SEED = 0
rng = np.random.default_rng(RNG_SEED)

n_weeks = 22
k_double_frozen = 11
k_single_frozen = 8

T_PERCEIVE_FIXED_SINGLE = 3.409  # must match polish_vetlyanka_fit_single_freeze_mu_gamma_FIXED.py
X0_X1_MIN_GAP = 2.0  # must match polish_vetlyanka_fit_double_freeze_mu_gamma_FIXED.py
X1_C1_PLACEHOLDER = 1.0
POLISH_MAXITER = 25  # confirmed sufficient at this scale for the sigmoidal pair

t_start, t_end, dt = 0, 22, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]

observed_weekly_deaths = np.array([
    3, 0, 2, 0, 1, 0, 2, 3, 0, 1, 7, 8, 7, 56, 169,
    54, 32, 19, 12, 0, 0, 0
], dtype=float)

IDX_GAMMA1, IDX_GAMMA2, IDX_MU1, IDX_MU2 = 6, 7, 8, 9
GAMMA_CONST_BOUND = (min(BOUNDS[IDX_GAMMA1][0], BOUNDS[IDX_GAMMA2][0]), max(BOUNDS[IDX_GAMMA1][1], BOUNDS[IDX_GAMMA2][1]))
MU_CONST_BOUND = (min(BOUNDS[IDX_MU1][0], BOUNDS[IDX_MU2][0]), max(BOUNDS[IDX_MU1][1], BOUNDS[IDX_MU2][1]))

REDUCED_DOUBLE_FROZEN_NAMES = ["b1", "b2", "x0", "x1", "c", "c1", "gamma_const", "mu_const",
                                "T_perceive", "sensitivity", "dispose_rate"]
REDUCED_DOUBLE_FROZEN_BOUNDS = [BOUNDS[0], BOUNDS[1], BOUNDS[2], BOUNDS[3], BOUNDS[4], BOUNDS[5],
                                  GAMMA_CONST_BOUND, MU_CONST_BOUND, BOUNDS[10], BOUNDS[11], BOUNDS[12]]
IDX_X0_RDF = REDUCED_DOUBLE_FROZEN_NAMES.index("x0")
IDX_X1_RDF = REDUCED_DOUBLE_FROZEN_NAMES.index("x1")

IDX_GAMMA1_S, IDX_GAMMA2_S, IDX_MU1_S, IDX_MU2_S = 4, 5, 6, 7
GAMMA_CONST_BOUND_S = (min(BOUNDS_SINGLE[IDX_GAMMA1_S][0], BOUNDS_SINGLE[IDX_GAMMA2_S][0]),
                        max(BOUNDS_SINGLE[IDX_GAMMA1_S][1], BOUNDS_SINGLE[IDX_GAMMA2_S][1]))
MU_CONST_BOUND_S = (min(BOUNDS_SINGLE[IDX_MU1_S][0], BOUNDS_SINGLE[IDX_MU2_S][0]),
                     max(BOUNDS_SINGLE[IDX_MU1_S][1], BOUNDS_SINGLE[IDX_MU2_S][1]))
REDUCED_SINGLE_FROZEN_NAMES = ["b1", "b2", "x0", "c", "gamma_const", "mu_const", "sensitivity", "dispose_rate"]
REDUCED_SINGLE_FROZEN_BOUNDS = [BOUNDS_SINGLE[0], BOUNDS_SINGLE[1], BOUNDS_SINGLE[2], BOUNDS_SINGLE[3],
                                  GAMMA_CONST_BOUND_S, MU_CONST_BOUND_S, BOUNDS_SINGLE[9], BOUNDS_SINGLE[10]]


# ------------------------------------------------------------
# Load libraries -- candidate_params_frozen_main.npy only covers the
# MAIN batch, so restrict starting-point search to those rows.
# ------------------------------------------------------------
candidate_params_main = np.load(os.path.join(BOOTSTRAP_DIR, "candidate_params_frozen_main.npy"))
n_main = candidate_params_main.shape[0]

double_frozen_trajectories = np.load(os.path.join(BOOTSTRAP_DIR, "double_frozen_trajectories.npy"))[:n_main]
single_frozen_trajectories = np.load(os.path.join(BOOTSTRAP_DIR, "single_frozen_trajectories.npy"))

double_valid = ~np.isnan(double_frozen_trajectories).any(axis=1)
single_valid = ~np.isnan(single_frozen_trajectories).any(axis=1)
double_traj_v = double_frozen_trajectories[double_valid]
double_params_v = candidate_params_main[double_valid]
single_traj_v = single_frozen_trajectories[single_valid]
single_params_v = candidate_params_main[single_valid]
print(f"Using {double_traj_v.shape[0]} valid double-frozen candidates, "
      f"{single_traj_v.shape[0]} valid single-frozen candidates (main batch only).")


# ------------------------------------------------------------
# Simulation + polish helpers
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


def double_frozen_reduced_to_full(reduced):
    (b1, b2, x0, x1, c, c1, gamma_const, mu_const, T_perceive, sensitivity, dispose_rate) = reduced
    return np.array([b1, b2, x0, x1, c, c1, gamma_const, gamma_const,
                      mu_const, mu_const, T_perceive, sensitivity, dispose_rate])


def sse_double_frozen(reduced, target):
    if reduced[IDX_X0_RDF] >= reduced[IDX_X1_RDF] - X0_X1_MIN_GAP:
        return 1e12
    full = double_frozen_reduced_to_full(reduced)
    predicted = simulate_weekly(plague_model_double, full)
    if predicted is None:
        return 1e12
    sse = np.sum((target - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def single_frozen_reduced_to_full(reduced):
    d = dict(zip(REDUCED_SINGLE_FROZEN_NAMES, reduced))
    single_dict = {
        "b1": d["b1"], "b2": d["b2"], "x0": d["x0"], "c": d["c"],
        "gamma1": d["gamma_const"], "gamma2": d["gamma_const"],
        "mu1": d["mu_const"], "mu2": d["mu_const"],
        "T_perceive": T_PERCEIVE_FIXED_SINGLE,
        "sensitivity": d["sensitivity"], "dispose_rate": d["dispose_rate"],
    }
    return np.array([single_dict.get(name, X1_C1_PLACEHOLDER) for name in PARAM_NAMES])


def sse_single_frozen(reduced, target):
    full = single_frozen_reduced_to_full(reduced)
    predicted = simulate_weekly(plague_model_single, full)
    if predicted is None:
        return 1e12
    sse = np.sum((target - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def scale_parameters(params):
    scaling_factors = np.maximum(np.abs(params), 1e-3)
    return params / scaling_factors, scaling_factors


def start_double_frozen_from_main(main_row):
    """candidate_params_frozen_main.npy stores samples drawn DIRECTLY in
    the 11-dim reduced double-frozen space (build_frozen_library.py's own
    design), matching REDUCED_DOUBLE_FROZEN_NAMES order exactly -- no
    conversion needed, use directly as the polish starting point."""
    return main_row.copy()


def start_single_frozen_from_main(main_row):
    """Extract single-frozen's 8 relevant values from the SAME 11-dim
    reduced double-frozen vector (b1,b2,x0,x1,c,c1,gamma_const,mu_const,
    T_perceive,sensitivity,dispose_rate) -- dropping x1,c1 (single has no
    decline-onset/decline-steepness parameters) and T_perceive (fixed
    separately, not part of either space's free search here)."""
    b1, b2, x0, x1, c, c1, gamma_const, mu_const, T_perceive, sensitivity, dispose_rate = main_row
    return np.array([b1, b2, x0, c, gamma_const, mu_const, sensitivity, dispose_rate])


def polish_double_frozen(start_full_row, target):
    start_reduced = start_double_frozen_from_main(start_full_row)
    scaled_x0, scaling_factors = scale_parameters(start_reduced)
    scaled_bounds = [(lo / sf, hi / sf) for (lo, hi), sf in zip(REDUCED_DOUBLE_FROZEN_BOUNDS, scaling_factors)]
    result = minimize(
        lambda sp: sse_double_frozen(sp * scaling_factors, target),
        scaled_x0, method="L-BFGS-B", bounds=scaled_bounds,
        options={"maxiter": POLISH_MAXITER, "ftol": 1e-10, "gtol": 1e-8},
    )
    optimized = result.x * scaling_factors
    return double_frozen_reduced_to_full(optimized)


def polish_single_frozen(start_full_row, target):
    start_reduced = start_single_frozen_from_main(start_full_row)
    scaled_x0, scaling_factors = scale_parameters(start_reduced)
    scaled_bounds = [(lo / sf, hi / sf) for (lo, hi), sf in zip(REDUCED_SINGLE_FROZEN_BOUNDS, scaling_factors)]
    result = minimize(
        lambda sp: sse_single_frozen(sp * scaling_factors, target),
        scaled_x0, method="L-BFGS-B", bounds=scaled_bounds,
        options={"maxiter": POLISH_MAXITER, "ftol": 1e-10, "gtol": 1e-8},
    )
    optimized = result.x * scaling_factors
    return single_frozen_reduced_to_full(optimized)


# ------------------------------------------------------------
# Ground truth: real single-frozen best fit, re-solved directly
# ------------------------------------------------------------
single_frozen_df = pd.read_csv(
    os.path.join(FITS_SINGLE_FROZEN_DIR, "vetlyanka_constant_mu_gamma_single_FIXED_fit.csv")
).set_index("Parameter")
b1, b2, x0, c = [single_frozen_df.loc[n, "Best_Fit"] for n in ["b1", "b2", "x0", "c"]]
gamma_const, mu_const = [single_frozen_df.loc[n, "Best_Fit"] for n in ["gamma_const", "mu_const"]]
sensitivity, dispose_rate = [single_frozen_df.loc[n, "Best_Fit"] for n in ["sensitivity", "dispose_rate"]]
truth_params = single_frozen_reduced_to_full([b1, b2, x0, c, gamma_const, mu_const, sensitivity, dispose_rate])

sol = solve_ivp(plague_model_single, [t_start, t_end], initial_conditions,
                 args=(truth_params,), t_eval=t_points, method="RK45")
true_cumulative = sol.y[5][::steps_per_week][:n_weeks]


def to_incremental(cumulative, baseline=0.0):
    cumulative = np.asarray(cumulative, dtype=float)
    return np.diff(np.concatenate(([baseline], cumulative)))


true_weekly_increments = np.clip(to_incremental(true_cumulative), 0, None)

# ------------------------------------------------------------
# Estimate negative-binomial overdispersion
#
# NB2 parameterization:
#     E[Y_t]   = mu_t
#     Var[Y_t] = mu_t + mu_t^2 / phi
#
# phi -> infinity corresponds to the Poisson limit.
# ------------------------------------------------------------

mu = np.clip(true_weekly_increments, 1e-12, None)
y = observed_weekly_deaths

numerator = np.sum(mu ** 2)
denominator = np.sum((y - mu) ** 2 - mu)

if denominator > 0:
    phi_hat = numerator / denominator
else:
    phi_hat = np.inf

print(f"\nEstimated NB dispersion phi = {phi_hat:.6f}")

if np.isfinite(phi_hat):
    print("Using negative-binomial bootstrap.")
else:
    print("No detectable extra-Poisson variation; NB reduces to Poisson.")

def calculate_information_criteria(observed, predicted, k):
    n = observed.size
    sse = float(np.sum((observed - predicted) ** 2))
    if sse <= 0:
        sse = np.finfo(float).tiny
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    aicc = np.inf if n <= k + 1 else aic + (2 * k * (k + 1)) / (n - k - 1)
    return aic, aicc, bic

def draw_negative_binomial(mu, phi, rng):
    """
    Draw counts under the NB2 parameterization:

        E[Y]   = mu
        Var[Y] = mu + mu^2 / phi

    NumPy parameterizes the negative binomial using (n, p), where

        mean = n * (1 - p) / p.

    Setting n = phi and p = phi / (phi + mu) gives the desired mean
    and variance.
    """
    mu = np.asarray(mu, dtype=float)

    if not np.isfinite(phi):
        return rng.poisson(mu).astype(float)

    p = phi / (phi + mu)

    return rng.negative_binomial(
        n=phi,
        p=p
    ).astype(float)
# ------------------------------------------------------------
# Bootstrap loop
# ------------------------------------------------------------
records = []

for b in range(N_BOOTSTRAP):

    synthetic_increments = draw_negative_binomial(
        true_weekly_increments,
        phi_hat,
        rng
    )

    synthetic_cumulative = np.cumsum(synthetic_increments)

    sse_d_lib = np.sum(
        (double_traj_v - synthetic_cumulative) ** 2,
        axis=1
    )
    sse_s_lib = np.sum(
        (single_traj_v - synthetic_cumulative) ** 2,
        axis=1
    )

    start_double = double_params_v[np.argmin(sse_d_lib)]
    start_single = single_params_v[np.argmin(sse_s_lib)]

    polished_double_full = polish_double_frozen(
        start_double,
        synthetic_cumulative
    )
    polished_single_full = polish_single_frozen(
        start_single,
        synthetic_cumulative
    )

    best_double = simulate_weekly(
        plague_model_double,
        polished_double_full
    )
    best_single = simulate_weekly(
        plague_model_single,
        polished_single_full
    )

    if best_double is None or best_single is None:
        continue

    aic_d, aicc_d, bic_d = calculate_information_criteria(
        synthetic_cumulative,
        best_double,
        k_double_frozen
    )
    aic_s, aicc_s, bic_s = calculate_information_criteria(
        synthetic_cumulative,
        best_single,
        k_single_frozen
    )

    synth_incr = to_incremental(synthetic_cumulative)

    aic_d_i, aicc_d_i, bic_d_i = calculate_information_criteria(
        synth_incr,
        to_incremental(best_double),
        k_double_frozen
    )
    aic_s_i, aicc_s_i, bic_s_i = calculate_information_criteria(
        synth_incr,
        to_incremental(best_single),
        k_single_frozen
    )

    records.append({
        "delta_AIC_cum": aic_s - aic_d,
        "delta_AICc_cum": aicc_s - aicc_d,
        "delta_BIC_cum": bic_s - bic_d,
        "delta_AIC_incr": aic_s_i - aic_d_i,
        "delta_AICc_incr": aicc_s_i - aicc_d_i,
        "delta_BIC_incr": bic_s_i - bic_d_i,
    })

    if (b + 1) % 50 == 0:
        print(f"  NB bootstrap replicate {b + 1}/{N_BOOTSTRAP}")

null_df = pd.DataFrame(records)
null_df.to_csv(
    os.path.join(
        BOOTSTRAP_DIR,
        "bootstrap_null_distribution_frozen_negbin.csv"
    ),
    index=False
)

# From AIC_AICc_BIC_gamma_mu_freeze.py's "Double-frozen minus Single-frozen"
# line, negated (this script's convention is single minus double).
observed = {
    "delta_AIC_cum": 11.499, "delta_AICc_cum": -3.824, "delta_BIC_cum": 8.226,
    "delta_AIC_incr": 11.197, "delta_AICc_incr": -4.126, "delta_BIC_incr": 7.924,
}
print("\nNull distribution summary (from single-frozen synthetic data):")
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
ax.set_xlabel("\u0394AICc (single-frozen \u2212 double-frozen), incremental basis")
ax.set_ylabel("Count across bootstrap replicates")
ax.set_title(
    "Null distribution under a true single-regime world\n"
    "negative-binomial observation noise"
)
ax.legend()
fig.tight_layout()
fig.savefig(
    os.path.join(
        BOOTSTRAP_DIR,
        "bootstrap_null_distribution_frozen_negbin.png"
    ),
    dpi=200
)
print(f"\nSaved {BOOTSTRAP_DIR}/bootstrap_null_distribution_frozen_negbin.png")