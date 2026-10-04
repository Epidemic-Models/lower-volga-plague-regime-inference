"""
polish_vetlyanka_fit_double_freeze_mu_gamma_FIXED.py   (step 02 -- run BEFORE the single-frozen fit)

Fit of the Vetlyanka double-sigmoid model with CONSTANT recovery and
mortality rates ("double-frozen" / "constant gamma, mu").

The full double-sigmoid model has 13 parameters:
    b1, b2, x0, x1, c, c1, gamma1, gamma2, mu1, mu2,
    T_perceive, sensitivity, dispose_rate
This model sets
    gamma1 = gamma2 = gamma_const,   mu1 = mu2 = mu_const
so it has 11 free parameters. gamma_const and mu_const are searched over
the SPAN (union) of the two original ranges.

Reads 01_sampling/parameter_samples_<NUM_SAMPLES>.csv and writes to
02_fits/results/double_frozen/. The single-frozen fit reads this model's
fitted T_perceive from that output, so run this script first.

PROCEDURE (deterministic -- the seeded sample file gives the same answer
every run): score every LHS sample, polish the TOP_K best with L-BFGS-B
(scaled parameters, box bounds), and keep the lowest polished SSE.
The SSE surface has many shallow local minima, so the result is the best
fit found by the multi-start search.

STARTING VALUES FOR gamma_const / mu_const: the shared sample file has
separate gamma1, gamma2 (and mu1, mu2) columns. Averaging them (the
previous approach) puts almost every starting value in the middle of the
span and almost none near its edges. Instead, the gamma1 (mu1) column's
relative position within its own range is mapped onto the span, so the
starting values are spread uniformly over the whole span (still one value
per LHS stratum). Set SPAN_STARTS = False to use the old averaging.

OBSERVATION TIMES: end-of-week cumulative deaths for weeks 1..22, so the
model is compared at t = 1..22. Set LEGACY_ALIGNMENT = True to reproduce
the published version (t = 0..21).

X0/X1 ORDERING CONSTRAINT: candidates with x1 - x0 < X0_X1_MIN_GAP are
rejected (2 weeks for this model, as in the published analysis).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # .../vetlyanka/02_fits
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):  # walk up to vetlyanka/, which contains src/
    if os.path.dirname(ROOT) == ROOT:
        raise FileNotFoundError("Could not find the src/ folder above this script.")
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from plague_double_sigmoid_model import plague_model
from vetlyanka_bounds import PARAM_NAMES, BOUNDS

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
LEGACY_ALIGNMENT = False   # True = published (t = 0..21); False = corrected (t = 1..22)
SPAN_STARTS = True         # True = starting gamma_const/mu_const spread over the whole span

SAMPLES_DIR = os.path.join(ROOT, "01_sampling")
OUTPUT_DIR = os.path.join(HERE, "results", "double_frozen")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 400000  # must match 01_sampling/parameter_samples_<NUM_SAMPLES>.csv
SAMPLES_PATH = os.path.join(SAMPLES_DIR, f"parameter_samples_{NUM_SAMPLES}.csv")
TOP_K = 20            # polish this many independent starting points, keep the best
X0_X1_MIN_GAP = 2.0   # require x1 - x0 >= this many weeks

# Extra search stage (see docstring)
USE_NESTED_START = False   # stage 2: also start from the double fit's optimum
DOUBLE_FIT_PATH = os.path.join(HERE, "results", "double", "vetlyanka_polished_fit.csv")

FIT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "vetlyanka_constant_mu_gamma_polished_fit.csv")
PREDICTIONS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "vetlyanka_constant_mu_gamma_predictions.csv")
SUMMARY_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "vetlyanka_constant_mu_gamma_summary.txt")
MULTISTART_LOG_PATH = os.path.join(OUTPUT_DIR, "vetlyanka_constant_mu_gamma_multistart_log.csv")

t_start, t_end, dt = 0.0, 22.0, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
n_weeks = 22
OBS = slice(0, n_weeks) if LEGACY_ALIGNMENT else slice(1, n_weeks + 1)
OBS_WEEKS = np.arange(0, n_weeks) if LEGACY_ALIGNMENT else np.arange(1, n_weeks + 1)
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]  # S0, I0, R0, DI0, DDI0, DR0, P0

# ---------------------------------------------------------------------
# Parameter structure
# ---------------------------------------------------------------------
EXPECTED_FULL_NAMES = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2",
                       "mu1", "mu2", "T_perceive", "sensitivity", "dispose_rate"]
if list(PARAM_NAMES) != EXPECTED_FULL_NAMES or len(BOUNDS) != 13:
    raise ValueError(f"Unexpected parameter order in vetlyanka_bounds.py: {list(PARAM_NAMES)}")

REDUCED_PARAM_NAMES = ["b1", "b2", "x0", "x1", "c", "c1",
                       "gamma_const", "mu_const", "T_perceive", "sensitivity", "dispose_rate"]


def span_bound(bound_1, bound_2):
    """Union of two ranges (smallest range containing both)."""
    return (min(bound_1[0], bound_2[0]), max(bound_1[1], bound_2[1]))


GAMMA_CONST_BOUND = span_bound(BOUNDS[6], BOUNDS[7])
MU_CONST_BOUND = span_bound(BOUNDS[8], BOUNDS[9])
REDUCED_BOUNDS = [BOUNDS[0], BOUNDS[1], BOUNDS[2], BOUNDS[3], BOUNDS[4], BOUNDS[5],
                  GAMMA_CONST_BOUND, MU_CONST_BOUND, BOUNDS[10], BOUNDS[11], BOUNDS[12]]
K_REDUCED = len(REDUCED_PARAM_NAMES)  # 11
IDX_X0, IDX_X1 = 2, 3


def load_observed_data():
    """Vetlyanka cumulative deaths at the end of weeks 1..22."""
    return np.array([3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19, 27, 34,
                     90, 259, 313, 345, 364, 376, 376, 376, 376], dtype=float)


def _to_span(value, own_bound, span):
    """Map a value's relative position in its own range onto the span."""
    u = (value - own_bound[0]) / (own_bound[1] - own_bound[0])
    return span[0] + u * (span[1] - span[0])


def full_to_reduced(full_params):
    """Turn a 13-column sample row into an 11-parameter starting point."""
    f = np.asarray(full_params, dtype=float)
    if SPAN_STARTS:
        gamma_const = _to_span(f[6], BOUNDS[6], GAMMA_CONST_BOUND)
        mu_const = _to_span(f[8], BOUNDS[8], MU_CONST_BOUND)
    else:
        gamma_const = 0.5 * (f[6] + f[7])
        mu_const = 0.5 * (f[8] + f[9])
    return np.array([f[0], f[1], f[2], f[3], f[4], f[5], gamma_const, mu_const,
                     f[10], f[11], f[12]], dtype=float)


def reduced_to_full(r):
    """Expand the 11-parameter vector to the 13 slots plague_model expects."""
    b1, b2, x0, x1, c, c1, g, m, T_perceive, sensitivity, dispose_rate = np.asarray(r, dtype=float)
    return np.array([b1, b2, x0, x1, c, c1, g, g, m, m, T_perceive, sensitivity, dispose_rate])


# ---------------------------------------------------------------------
# Simulation and objective
# ---------------------------------------------------------------------
def simulate_cumulative(reduced_params):
    try:
        sol = solve_ivp(plague_model, [t_start, t_end], initial_conditions,
                        args=(reduced_to_full(reduced_params),), t_eval=t_points, method="RK45")
    except (ValueError, FloatingPointError, OverflowError):
        return None
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    predicted = sol.y[5][::steps_per_week][OBS]  # DR = cumulative deaths at the observation times
    return predicted if predicted.shape[0] == n_weeks else None


def sse_unscaled(reduced_params, observed):
    if reduced_params[IDX_X0] >= reduced_params[IDX_X1] - X0_X1_MIN_GAP:
        return 1e12
    predicted = simulate_cumulative(reduced_params)
    if predicted is None:
        return 1e12
    sse = float(np.sum((observed - predicted) ** 2))
    return sse if np.isfinite(sse) else 1e12


def sse_objective(scaled_params, scaling_factors, observed):
    return sse_unscaled(scaled_params * scaling_factors, observed)


def scale_parameters(params):
    scaling_factors = np.maximum(np.abs(params), 1e-3)
    return params / scaling_factors, scaling_factors


def find_top_k_lhs_candidates(samples_path, observed, k=TOP_K):
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"{samples_path} not found -- run 01_sampling/parameter_sampling.py first.")
    full_samples = np.loadtxt(samples_path, delimiter=",", skiprows=1)
    if full_samples.ndim != 2 or full_samples.shape[1] != 13:
        raise ValueError(f"Expected 13 columns in {samples_path}.")
    print(f"Scanning {full_samples.shape[0]} LHS candidates "
          f"({'span-spread' if SPAN_STARTS else 'averaged'} gamma_const/mu_const starts)...")
    scored = []
    for i, full_row in enumerate(full_samples):
        reduced_row = full_to_reduced(full_row)
        scored.append((sse_unscaled(reduced_row, observed), reduced_row))
        if (i + 1) % 20000 == 0:
            print(f"  scanned {i + 1}/{full_samples.shape[0]} "
                  f"(best so far: {min(s for s, _ in scored):.3f})", flush=True)
    scored.sort(key=lambda pair: pair[0])
    top_k = scored[:k]
    print(f"\nTop {k} LHS candidate SSEs: {[f'{s:.3f}' for s, _ in top_k]}")
    return top_k


def polish(start_row, observed):
    """One L-BFGS-B polish from start_row (unscaled); returns (sse, params, result, scaling)."""
    start_row = np.clip(start_row, [b[0] for b in REDUCED_BOUNDS], [b[1] for b in REDUCED_BOUNDS])
    scaled_x0, scaling_factors = scale_parameters(start_row)
    scaled_bounds = [(lo / sf, hi / sf) for (lo, hi), sf in zip(REDUCED_BOUNDS, scaling_factors)]
    result = minimize(sse_objective, scaled_x0, args=(scaling_factors, observed),
                      method="L-BFGS-B", bounds=scaled_bounds,
                      options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 100})
    p = result.x * scaling_factors
    return sse_unscaled(p, observed), p, result, scaling_factors


def nested_start_from_double():
    """The double-frozen model is the double model with gamma1=gamma2, mu1=mu2.
    Start from the double fit's optimum with gamma and mu averaged (None if not available)."""
    if not os.path.exists(DOUBLE_FIT_PATH):
        print(f"(No double fit at {DOUBLE_FIT_PATH} -- skipping the nested start.)")
        return None
    fit = pd.read_csv(DOUBLE_FIT_PATH).set_index("Parameter")["Best_Fit"]
    full = np.array([fit[name] for name in EXPECTED_FULL_NAMES], dtype=float)
    f = full.copy()
    return np.array([f[0], f[1], f[2], f[3], f[4], f[5], 0.5 * (f[6] + f[7]), 0.5 * (f[8] + f[9]),
                     f[10], f[11], f[12]], dtype=float)


def information_criteria(sse, n, k):
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    aicc = aic + 2 * k * (k + 1) / (n - k - 1)
    return aic, aicc, bic


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    observed = load_observed_data()
    print("Constant-gamma/constant-mu (double-frozen) Vetlyanka fit")
    print(f"Observation times: {'t = 0..21 (LEGACY)' if LEGACY_ALIGNMENT else 't = 1..22 (end of each week)'}")
    print(f"Reading samples from: {SAMPLES_PATH}\nWriting outputs to:   {OUTPUT_DIR}\n")
    for name, bound in zip(REDUCED_PARAM_NAMES, REDUCED_BOUNDS):
        print(f"  {name:15s} {bound}")

    top_candidates = find_top_k_lhs_candidates(SAMPLES_PATH, observed)

    # Stage 1: polish the TOP_K best LHS candidates
    best = None   # (sse, params, result, scaling)
    multistart_log = []

    def record(stage, label, start_sse, out):
        nonlocal best
        sse, p, result, _ = out
        print(f"  -> polished SSE={sse:.6f}  (iters={result.nit}, x0={p[IDX_X0]:.3f}, x1={p[IDX_X1]:.3f}, "
              f"T_perceive={p[8]:.3f})", flush=True)
        multistart_log.append({"stage": stage, "start": label, "start_sse": start_sse, "polished_sse": sse,
                               "iterations": result.nit, "evaluations": result.nfev,
                               "x0": p[IDX_X0], "x1": p[IDX_X1], "T_perceive": p[8]})
        if best is None or sse < best[0]:
            best = out

    for rank, (lhs_sse, start_row) in enumerate(top_candidates):
        print(f"\n[stage 1] LHS start #{rank + 1}/{len(top_candidates)} (LHS SSE={lhs_sse:.3f})...", flush=True)
        record("1_lhs", f"lhs_{rank + 1}", lhs_sse, polish(start_row, observed))
    stage1_sse = sorted(r["polished_sse"] for r in multistart_log)
    n_agree = sum(s < stage1_sse[0] * 1.001 for s in stage1_sse)
    print(f"\nStage 1 best SSE = {stage1_sse[0]:.6f}; {n_agree} of {len(stage1_sse)} LHS starts reached it.")

    # Stage 2: nested start from the double fit (gamma and mu averaged)
    if USE_NESTED_START:
        nested = nested_start_from_double()
        if nested is not None:
            print("\n[stage 2] Start from the double fit's optimum (gamma, mu averaged)...", flush=True)
            record("2_nested", "double_fit", sse_unscaled(nested, observed), polish(nested, observed))

    pd.DataFrame(multistart_log).to_csv(MULTISTART_LOG_PATH, index=False)
    best_row = min(multistart_log, key=lambda r: r["polished_sse"])
    print(f"\nBest overall: {best_row['start']} (stage {best_row['stage']}), SSE = {best[0]:.6f}")
    print(f"Improvement over stage 1: {stage1_sse[0] - best[0]:.3f}")
    _, _, best_result, best_scaling_factors = best

    optimized = best_result.x * best_scaling_factors
    polished_sse = sse_unscaled(optimized, observed)
    predicted = simulate_cumulative(optimized)
    aic, aicc, bic = information_criteria(polished_sse, n_weeks, K_REDUCED)

    # Inverse-Hessian standard errors are a rough approximation and NOT reliable
    # intervals -- kept for diagnostics only; use 07_uncertainty for reporting.
    try:
        hess_inv = best_result.hess_inv
        hess_inv = hess_inv if isinstance(hess_inv, np.ndarray) else hess_inv.todense()
        cov = np.asarray(hess_inv) * np.outer(best_scaling_factors, best_scaling_factors)
        std_errors = np.sqrt(np.clip(np.diag(cov), 0, None))
    except (AttributeError, ValueError) as e:
        print(f"\nHessian inverse not usable ({e}).")
        std_errors = np.full(K_REDUCED, np.nan)

    fit_df = pd.DataFrame({
        "Parameter": REDUCED_PARAM_NAMES,
        "Bound_lo": [b[0] for b in REDUCED_BOUNDS],
        "Bound_hi": [b[1] for b in REDUCED_BOUNDS],
        "Best_Fit": optimized,
        "SE_hessian_approx_unreliable": std_errors,
    })
    fit_df.to_csv(FIT_OUTPUT_PATH, index=False)
    pd.DataFrame({"Week": OBS_WEEKS, "Observed_cumulative_deaths": observed,
                  "Predicted_cumulative_deaths": predicted,
                  "Residual": observed - predicted}).to_csv(PREDICTIONS_OUTPUT_PATH, index=False)

    gap = optimized[IDX_X1] - optimized[IDX_X0]
    near_bound = [name for name, (lo, hi), val in zip(REDUCED_PARAM_NAMES, REDUCED_BOUNDS, optimized)
                  if (val - lo) < 0.05 * (hi - lo) or (hi - val) < 0.05 * (hi - lo)]
    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(f"SSE={polished_sse:.6f}\nk={K_REDUCED}\nAIC={aic:.4f}\nAICc={aicc:.4f}\nBIC={bic:.4f}\n"
                 f"alignment={'legacy t=0..21' if LEGACY_ALIGNMENT else 't=1..22'}\n"
                 f"samples={os.path.basename(SAMPLES_PATH)}\nTOP_K={TOP_K}\nSPAN_STARTS={SPAN_STARTS}\n"
                 f"x1-x0={gap:.4f} (min gap {X0_X1_MIN_GAP})\n"
                 f"stage-1 LHS starts reaching stage-1 best: {n_agree}/{len(stage1_sse)}\n"
                 f"stage-1 best SSE: {stage1_sse[0]:.6f}; final best from {best_row['start']}\n"
                 f"within 5% of a bound: {near_bound}\n")

    print("\n", fit_df.to_string(index=False))
    print(f"\nSSE={polished_sse:.6f}  AIC={aic:.3f}  AICc={aicc:.3f}  BIC={bic:.3f}  (k={K_REDUCED})")
    print(f"x0={optimized[IDX_X0]:.3f}, x1={optimized[IDX_X1]:.3f}, gap={gap:.3f} weeks")
    if gap < X0_X1_MIN_GAP + 0.05:
        print(f"WARNING: x1 - x0 is on the minimum-gap constraint ({X0_X1_MIN_GAP}) -- "
              f"the constraint is active and the result depends on it.")
    print(f"\nWithin 5% of a bound: {near_bound or 'none'}")
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()