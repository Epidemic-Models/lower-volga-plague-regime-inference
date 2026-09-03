"""
polish_vetlyanka_fit_single_FIXED.py

Fit for the Vetlyanka SIGMOIDAL single-sigmoid model (rise-only beta(t),
with behavioral feedback -- gamma1/gamma2/mu1/mu2 remain independently
fitted, NOT frozen).

T_PERCEIVE FIXED, NOT SEARCHED: across three genuine multi-start runs at
successive ceilings (10, then 20), T_perceive pinned at the wall every
time, with SSE continuing to improve substantially each widening (763->
633) -- unlike single-frozen's flat/worsening pattern, this looked at
first like a real under-bounded parameter. But widening to 20 also
pulled dispose_rate from an interior value (3.26) down to its exact
floor (2.0) for the first time -- the same joint-escape signature found
and fixed in the double-sigmoid model's x0/x1 overlap: two parameters
moving TOGETHER into a degenerate configuration (here, an extremely long
perception lag paired with the slowest-possible corpse disposal, jointly
faking a "everything about this outbreak was slow" suppression effect
this model shouldn't be able to produce without a real regime shift),
not one parameter that genuinely needs more room. Confirmed by the same
pattern already seen in single-frozen (T_perceive high, dispose_rate low,
together) -- this is a limitation of single-regime models generally, not
specific to the frozen-gamma/mu variant.

T_perceive is fixed here at 3.432, double-sigmoid's own genuinely-
converged value (polish_vetlyanka_fit_double.py) -- the cleanest
available anchor, since double never needed fixing at all: it settled
there under ordinary multi-start, comfortably interior, with no
correlated pinning alongside it, unlike every value borrowed from a
model exhibiting the same symptom being explained away.

REQUIRED BEFORE RUNNING:
  1. Set NUM_SAMPLES below to match an existing parameter_samples_<N>.csv.
  2. If that file doesn't exist, or vetlyanka_bounds.py changed since it
     was generated, run parameter_sampling.py first.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from plague_single_sigmoid_model import plague_model
from vetlyanka_bounds import PARAM_NAMES, PARAM_NAMES_SINGLE, BOUNDS_SINGLE

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "fits", "single")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 200000  # must match an existing data/samples/parameter_samples_<NUM_SAMPLES>.csv
SAMPLES_PATH = os.path.join(SAMPLES_DIR, f"parameter_samples_{NUM_SAMPLES}.csv")
TOP_K = 10  # polish this many independent starting points, keep the best
MULTISTART_LOG_PATH = os.path.join(OUTPUT_DIR, "vetlyanka_single_FIXED_multistart_log.csv")
X1_C1_PLACEHOLDER = 1.0  # inert for the single-sigmoid model

T_PERCEIVE_FIXED = 3.571  # see module docstring -- borrowed from double-sigmoid's
                           # own genuinely-converged value, not searched here

t_start, t_end, dt = 0, 22, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
n_weeks = 22
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]

IDX_T_PERCEIVE_SINGLE = PARAM_NAMES_SINGLE.index("T_perceive")

# Indices of the single-sigmoid columns within the shared 13-column sample
# file, EXCLUDING T_perceive (fixed, not searched) as well as x1/c1.
_FULL_TO_SINGLE_IDX = [i for i, name in enumerate(PARAM_NAMES)
                        if name in PARAM_NAMES_SINGLE and name != "T_perceive"]

REDUCED_PARAM_NAMES = [name for name in PARAM_NAMES_SINGLE if name != "T_perceive"]
REDUCED_BOUNDS = [b for name, b in zip(PARAM_NAMES_SINGLE, BOUNDS_SINGLE) if name != "T_perceive"]
K_REDUCED = len(REDUCED_PARAM_NAMES)


def load_observed_data():
    return np.array([3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19, 27, 34,
                      90, 259, 313, 345, 364, 376, 376, 376, 376], dtype=float)


def expand_reduced_to_full(reduced_params):
    reduced_dict = dict(zip(REDUCED_PARAM_NAMES, reduced_params))
    reduced_dict["T_perceive"] = T_PERCEIVE_FIXED
    return np.array([reduced_dict.get(name, X1_C1_PLACEHOLDER) for name in PARAM_NAMES], dtype=float)


def simulate_cumulative(reduced_params):
    full_params = expand_reduced_to_full(reduced_params)
    try:
        sol = solve_ivp(plague_model, [t_start, t_end], initial_conditions,
                         args=(full_params,), t_eval=t_points, method="RK45")
    except (ValueError, FloatingPointError, OverflowError):
        return None
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    predicted = sol.y[5][::steps_per_week][:n_weeks]
    if predicted.shape[0] != n_weeks or not np.all(np.isfinite(predicted)):
        return None
    return predicted


def sse_unscaled(reduced_params, observed):
    predicted = simulate_cumulative(reduced_params)
    if predicted is None:
        return 1e12
    sse = np.sum((observed - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def sse_objective(scaled_params, scaling_factors, observed):
    return sse_unscaled(scaled_params * scaling_factors, observed)


def scale_parameters(params):
    scaling_factors = np.maximum(np.abs(params), 1e-3)
    return params / scaling_factors, scaling_factors


def calculate_aic(sse, n, k):
    return float(n * np.log(sse / n) + 2 * k) if sse > 0 and np.isfinite(sse) else np.nan


def calculate_aicc(sse, n, k):
    aic = calculate_aic(sse, n, k)
    denom = n - k - 1
    return float(aic + (2 * k * (k + 1)) / denom) if np.isfinite(aic) and denom > 0 else np.nan


def find_top_k_lhs_candidates(samples_path, observed, k=TOP_K):
    """Load the shared 13-column sample file, extract the 10 reduced-model
    columns from each row (11 single-sigmoid columns minus T_perceive),
    score, and return the top k."""
    full_samples = np.loadtxt(samples_path, delimiter=",", skiprows=1)
    print(f"Scanning {full_samples.shape[0]} shared LHS candidates "
          f"(T_perceive fixed at {T_PERCEIVE_FIXED}, not searched)...")

    scored = []
    for i, full_row in enumerate(full_samples):
        reduced_row = full_row[_FULL_TO_SINGLE_IDX]
        sse = sse_unscaled(reduced_row, observed)
        scored.append((sse, reduced_row))
        if (i + 1) % 20000 == 0:
            best_so_far = min(s for s, _ in scored)
            print(f"  scanned {i + 1}/{full_samples.shape[0]} (best so far: {best_so_far:.3f})")

    scored.sort(key=lambda pair: pair[0])
    top_k = scored[:k]
    print(f"\nTop {k} LHS candidate SSEs: {[f'{s:.3f}' for s, _ in top_k]}")
    return top_k


def main():
    observed = load_observed_data()

    top_candidates = find_top_k_lhs_candidates(SAMPLES_PATH, observed)

    best_result = None
    best_polished_sse = np.inf
    best_scaling_factors = None
    multistart_log = []

    for rank, (lhs_sse, start_row) in enumerate(top_candidates):
        scaled_x0, scaling_factors = scale_parameters(start_row)
        scaled_bounds = [(lo / sf, hi / sf) for (lo, hi), sf in zip(REDUCED_BOUNDS, scaling_factors)]

        print(f"\nPolishing start #{rank + 1}/{TOP_K} (LHS SSE={lhs_sse:.3f})...")
        result = minimize(sse_objective, scaled_x0, args=(scaling_factors, observed),
                           method="L-BFGS-B", bounds=scaled_bounds,
                           options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 100})

        optimized = result.x * scaling_factors
        polished_sse = sse_unscaled(optimized, observed)
        print(f"  -> polished SSE={polished_sse:.6f}  (iters={result.nit}, evals={result.nfev})")

        multistart_log.append({
            "rank": rank + 1, "lhs_sse": lhs_sse, "polished_sse": polished_sse,
            "iterations": result.nit, "evaluations": result.nfev,
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
        print("NOTE: the single-best-LHS-candidate start was NOT the best after polishing.")

    optimized_params = best_result.x * best_scaling_factors
    polished_sse = sse_unscaled(optimized_params, observed)
    aic = calculate_aic(polished_sse, n_weeks, K_REDUCED)
    aicc = calculate_aicc(polished_sse, n_weeks, K_REDUCED)

    print(f"\nPolished SSE: {polished_sse:.6f}")
    print(f"AIC: {aic:.6f}   AICc: {aicc:.6f}")
    print(f"(T_perceive fixed at {T_PERCEIVE_FIXED}, not counted as a free parameter -- K_REDUCED={K_REDUCED})")

    df = pd.DataFrame({
        "Parameter": REDUCED_PARAM_NAMES,
        "Bound_lo": [b[0] for b in REDUCED_BOUNDS],
        "Bound_hi": [b[1] for b in REDUCED_BOUNDS],
        "Best_Fit": optimized_params,
    })
    print("\n", df.to_string(index=False))
    df.to_csv(os.path.join(OUTPUT_DIR, "vetlyanka_polished_fit_single.csv"), index=False)
    print("\nSaved vetlyanka_polished_fit_single.csv")
    print("NOTE: T_perceive will NOT appear as a row in this CSV -- it is fixed, not fitted. "
          "Downstream scripts reading T_perceive from this file (AIC_AICc_BIC.py, "
          "parametric_bootstrap_test.py) need updating to use the fixed constant instead.")

    near_bound_flags = []
    for name, (lo, hi), val in zip(REDUCED_PARAM_NAMES, REDUCED_BOUNDS, optimized_params):
        span = hi - lo
        if (val - lo) < 0.05 * span or (hi - val) < 0.05 * span:
            near_bound_flags.append(name)
    if near_bound_flags:
        print(f"\nWARNING: still within 5% of a bound for: {near_bound_flags}")
    else:
        print("\nNo fitted parameters are within 5% of a bound.")


if __name__ == "__main__":
    main()