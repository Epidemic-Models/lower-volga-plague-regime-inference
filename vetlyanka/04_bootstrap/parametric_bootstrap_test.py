"""
parametric_bootstrap_test.py   (step 04 -- run after precompute_trajectory_library.py,
                                 expand_double_library_for_null_test.py and 03_model_comparison)

Parametric bootstrap (simulated-null) test for the SIGMOIDAL pair:
double-sigmoid vs single-sigmoid (time-varying gamma, mu). See
bootstrap_frozen.py for the constant-gamma/mu pair.

QUESTION: if the true process had only ONE regime (the fitted single-sigmoid
model), how often would the fitting procedure still favour the double
sigmoid by as much as it does on the real data?

PROCEDURE, per replicate:
  1. Synthetic data: Poisson noise on the weekly increments of the fitted
     single-sigmoid model (ground truth), cumulated.
  2. Starting points: the best-matching candidate in each model's precomputed
     trajectory library.
  3. Polish: one L-BFGS-B run (POLISH_MAXITER iterations) from that start, for
     BOTH models, within the same bounds and constraints as the real fits
     (single: T_perceive fixed; double: x1 - x0 >= X0_X1_MIN_GAP, x1 <= 20).
     The polish is what makes the null distribution use the same kind of
     fitting procedure as the observed deltas (see the history note below).
  4. Record delta = single - double for AIC, AICc, BIC on cumulative and
     incremental bases.
p-value = (1 + #{null delta >= observed delta}) / (1 + N), the standard
Monte-Carlo estimate that never returns exactly 0.

WIDENED-x1 CANDIDATES: the double library includes extra rows with x1 up to 40
(from expand_double_library_for_null_test.py) so a "no decline inside the
window" curve can be the starting point. The polish itself stays within the
REAL fit bounds (x1 <= 20), because the observed deltas come from fits with
those bounds; such starts are clipped into the bounds before polishing.

EVERYTHING IS READ, NOT TYPED IN:
  * observed deltas  <- 03_model_comparison/results/vetlyanka_pairwise_deltas.csv
  * ground truth      <- 02_fits/results/single/vetlyanka_polished_fit_single.csv
                         (including its fixed T_perceive)
  * library settings  <- 04_bootstrap/results/sigmoidal/library_info.json
The script stops if the library's T_perceive or alignment does not match the fits.

History: an earlier version compared UNPOLISHED library look-ups against
fully polished observed deltas, which handicapped single more than double
and inflated the null; the polish step fixes that.

Writes to 04_bootstrap/results/sigmoidal/:
    bootstrap_null_distribution.csv     -- one row per replicate
    bootstrap_pvalues.csv               -- observed delta, null summary, p-value per criterion
    bootstrap_null_distribution.png     -- histogram of the cumulative-basis delta AICc
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

import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from plague_double_sigmoid_model import plague_model as plague_model_double
from plague_single_sigmoid_model import plague_model as plague_model_single
from vetlyanka_bounds import PARAM_NAMES, PARAM_NAMES_SINGLE, BOUNDS, BOUNDS_SINGLE

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
LEGACY_ALIGNMENT = False   # must match the fits and the library
N_BOOTSTRAP = 500
RNG_SEED = 0
POLISH_MAXITER = 200     # tested at 60 and 300 with no meaningful change in the null distribution
X0_X1_MIN_GAP = 1.0        # must match polish_vetlyanka_fit_double.py
K_SINGLE, K_DOUBLE = 10, 13

LIB_DIR = os.path.join(HERE, "results", "sigmoidal")
SINGLE_FIT_PATH = os.path.join(ROOT, "02_fits", "results", "single", "vetlyanka_polished_fit_single.csv")
DELTAS_PATH = os.path.join(ROOT, "03_model_comparison", "results", "vetlyanka_pairwise_deltas.csv")
OBSERVED_COMPARISON = "Single minus Double"

t_start, t_end, dt = 0.0, 22.0, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
n_weeks = 22
OBS = slice(0, n_weeks) if LEGACY_ALIGNMENT else slice(1, n_weeks + 1)
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]
X1_C1_PLACEHOLDER = 1.0

SINGLE_FREE = [n for n in PARAM_NAMES_SINGLE if n != "T_perceive"]          # 10 names
SINGLE_FREE_BOUNDS = [b for n, b in zip(PARAM_NAMES_SINGLE, BOUNDS_SINGLE) if n != "T_perceive"]
FULL_TO_SINGLE_IDX = [PARAM_NAMES.index(n) for n in SINGLE_FREE]
IDX_X0, IDX_X1 = PARAM_NAMES.index("x0"), PARAM_NAMES.index("x1")
LO_D = np.array([b[0] for b in BOUNDS], float); HI_D = np.array([b[1] for b in BOUNDS], float)
LO_S = np.array([b[0] for b in SINGLE_FREE_BOUNDS], float); HI_S = np.array([b[1] for b in SINGLE_FREE_BOUNDS], float)


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------
def require(path, hint):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- {hint}")
    return path


def load_inputs():
    with open(require(os.path.join(LIB_DIR, "library_info.json"), "run precompute_trajectory_library.py first.")) as fh:
        info = json.load(fh)
    if info.get("legacy_alignment") != LEGACY_ALIGNMENT:
        raise ValueError("Library alignment does not match LEGACY_ALIGNMENT -- rebuild the library.")

    fit = pd.read_csv(require(SINGLE_FIT_PATH, "run 02_fits/polish_vetlyanka_fit_single_FIXED.py first."))
    fit = dict(zip(fit["Parameter"], fit["Best_Fit"].astype(float)))
    t_perceive = fit["T_perceive"]
    if abs(t_perceive - info["T_perceive_fixed_single"]) > 1e-9:
        raise ValueError(f"Library was built with T_perceive={info['T_perceive_fixed_single']} but the single "
                         f"fit uses {t_perceive} -- re-run precompute_trajectory_library.py.")

    double_traj = np.load(os.path.join(LIB_DIR, "double_trajectories.npy"))
    single_traj = np.load(os.path.join(LIB_DIR, "single_trajectories.npy"))
    params_main = np.load(os.path.join(LIB_DIR, "candidate_params.npy"))
    if info.get("widened_x1_appended"):
        params_wide = np.load(require(os.path.join(LIB_DIR, "candidate_params_widened_x1.npy"),
                                      "re-run expand_double_library_for_null_test.py."))
        double_params = np.vstack([params_main, params_wide])
    else:
        print("NOTE: widened-x1 batch not found -- run expand_double_library_for_null_test.py for the full test.")
        double_params = params_main
    if double_params.shape[0] != double_traj.shape[0] or params_main.shape[0] != single_traj.shape[0]:
        raise ValueError("Library rows and parameter rows do not line up -- rebuild the library.")

    deltas = pd.read_csv(require(DELTAS_PATH, "run 03_model_comparison/AIC_AICc_BIC_gamma_mu_freeze.py first."))
    observed = {}
    for basis, tag in (("cumulative", "cum"), ("incremental", "incr")):
        row = deltas[(deltas["basis"] == basis) & (deltas["comparison"] == OBSERVED_COMPARISON)].iloc[0]
        for crit in ("AIC", "AICc", "BIC"):
            observed[f"delta_{crit}_{tag}"] = float(row[f"d{crit}"])
    return info, fit, t_perceive, double_traj, single_traj, double_params, params_main, observed


# ---------------------------------------------------------------------
# Simulation, objective, polish (mirror the real fitting scripts)
# ---------------------------------------------------------------------
def simulate_weekly(model_fn, full_params):
    try:
        sol = solve_ivp(model_fn, [t_start, t_end], initial_conditions,
                        args=(full_params,), t_eval=t_points, method="RK45")
    except (ValueError, FloatingPointError, OverflowError):
        return None
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    weekly = sol.y[5][::steps_per_week][OBS]
    return weekly if len(weekly) == n_weeks else None


def scale_parameters(params):
    sf = np.maximum(np.abs(params), 1e-3)
    return params / sf, sf


def make_single_full(t_perceive):
    def single_full(reduced):
        d = dict(zip(SINGLE_FREE, reduced)); d["T_perceive"] = t_perceive
        return np.array([d.get(n, X1_C1_PLACEHOLDER) for n in PARAM_NAMES], dtype=float)
    return single_full


def sse(model_fn, full, target):
    pred = simulate_weekly(model_fn, full)
    if pred is None:
        return 1e12
    v = float(np.sum((target - pred) ** 2))
    return v if np.isfinite(v) else 1e12


def sse_double(full, target):
    if full[IDX_X0] >= full[IDX_X1] - X0_X1_MIN_GAP:
        return 1e12
    return sse(plague_model_double, full, target)


def polish(objective, start, lo, hi):
    start = np.clip(start, lo, hi)
    x0, sf = scale_parameters(start)
    res = minimize(lambda p: objective(p * sf), x0, method="L-BFGS-B",
                   bounds=list(zip(lo / sf, hi / sf)),
                   options={"maxiter": POLISH_MAXITER, "ftol": 1e-10, "gtol": 1e-8})
    return res.x * sf


def information_criteria(observed, predicted, k):
    n = observed.size
    s = max(float(np.sum((observed - predicted) ** 2)), np.finfo(float).tiny)
    aic = n * np.log(s / n) + 2 * k
    bic = n * np.log(s / n) + k * np.log(n)
    aicc = np.inf if n <= k + 1 else aic + 2 * k * (k + 1) / (n - k - 1)
    return aic, aicc, bic


def to_incremental(cumulative, baseline=0.0):
    return np.diff(np.concatenate(([baseline], np.asarray(cumulative, dtype=float))))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    info, fit, t_perceive, double_traj, single_traj, double_params, single_params, observed = load_inputs()
    single_full = make_single_full(t_perceive)

    dv = ~np.isnan(double_traj).any(axis=1); sv = ~np.isnan(single_traj).any(axis=1)
    double_traj_v, double_params_v = double_traj[dv], double_params[dv]
    single_traj_v, single_params_v = single_traj[sv], single_params[sv]

    truth_full = single_full(np.array([fit[n] for n in SINGLE_FREE]))
    true_cumulative = simulate_weekly(plague_model_single, truth_full)
    true_increments = np.clip(to_incremental(true_cumulative), 0, None)

    print(f"Observation times: {'t = 0..21 (LEGACY)' if LEGACY_ALIGNMENT else 't = 1..22 (end of each week)'}")
    print(f"Library: {double_traj_v.shape[0]} valid double candidates "
          f"({double_traj.shape[0]} rows incl. widened-x1), {single_traj_v.shape[0]} valid single candidates")
    print(f"Ground truth: single-sigmoid fit, T_perceive fixed at {t_perceive:.6f}")
    print(f"Observed deltas ({OBSERVED_COMPARISON}, from 03_model_comparison): "
          + ", ".join(f"{k}={v:.3f}" for k, v in observed.items()))
    print(f"Running {N_BOOTSTRAP} replicates (seed {RNG_SEED})...\n")

    rng = np.random.default_rng(RNG_SEED)
    records, n_failed, t0 = [], 0, time.time()
    for b in range(N_BOOTSTRAP):
        synth_cum = np.cumsum(rng.poisson(true_increments).astype(float))

        start_d = double_params_v[np.argmin(np.sum((double_traj_v - synth_cum) ** 2, axis=1))]
        start_s = single_params_v[np.argmin(np.sum((single_traj_v - synth_cum) ** 2, axis=1))]

        pol_d = polish(lambda p: sse_double(p, synth_cum), start_d, LO_D, HI_D)
        pol_s = single_full(polish(lambda p: sse(plague_model_single, single_full(p), synth_cum),
                                   start_s[FULL_TO_SINGLE_IDX], LO_S, HI_S))
        best_d = simulate_weekly(plague_model_double, pol_d)
        best_s = simulate_weekly(plague_model_single, pol_s)
        if best_d is None or best_s is None:
            n_failed += 1
            continue

        aic_d, aicc_d, bic_d = information_criteria(synth_cum, best_d, K_DOUBLE)
        aic_s, aicc_s, bic_s = information_criteria(synth_cum, best_s, K_SINGLE)
        si = to_incremental(synth_cum)
        aic_di, aicc_di, bic_di = information_criteria(si, to_incremental(best_d), K_DOUBLE)
        aic_si, aicc_si, bic_si = information_criteria(si, to_incremental(best_s), K_SINGLE)
        records.append({"replicate": b + 1,
                        "delta_AIC_cum": aic_s - aic_d, "delta_AICc_cum": aicc_s - aicc_d, "delta_BIC_cum": bic_s - bic_d,
                        "delta_AIC_incr": aic_si - aic_di, "delta_AICc_incr": aicc_si - aicc_di,
                        "delta_BIC_incr": bic_si - bic_di})
        if (b + 1) % 25 == 0:
            rate = (b + 1) / (time.time() - t0)
            print(f"  replicate {b + 1}/{N_BOOTSTRAP}  (~{(N_BOOTSTRAP - b - 1) / rate / 60:.0f} min left)", flush=True)

    null_df = pd.DataFrame(records)
    null_df.to_csv(os.path.join(LIB_DIR, "bootstrap_null_distribution.csv"), index=False)
    n = len(null_df)
    print(f"\nCompleted replicates: {n} (failed solves skipped: {n_failed})")

    rows = []
    for col, obs_val in observed.items():
        vals = null_df[col].values
        n_ge = int(np.sum(vals >= obs_val))
        rows.append({"statistic": col, "observed": obs_val, "null_mean": vals.mean(), "null_sd": vals.std(ddof=1),
                     "null_95th_pct": np.percentile(vals, 95), "null_max": vals.max(),
                     "n_null_ge_observed": n_ge, "n_replicates": n,
                     "p_value": (1 + n_ge) / (1 + n), "p_value_plain_fraction": n_ge / n})
    pvals = pd.DataFrame(rows)
    pvals.to_csv(os.path.join(LIB_DIR, "bootstrap_pvalues.csv"), index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
    print("\nObserved delta vs null distribution (positive delta favours double):")
    print(pvals.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_df["delta_AICc_cum"], bins=25, color="steelblue", edgecolor="white")
    ax.axvline(observed["delta_AICc_cum"], color="crimson", linewidth=2, label="Observed (real data)")
    ax.set_xlabel("ΔAICc (single − double), cumulative basis")
    ax.set_ylabel("Bootstrap replicates")
    ax.set_title("Null distribution: data simulated from the single-regime fit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(LIB_DIR, "bootstrap_null_distribution.png"), dpi=200)
    print(f"\nSaved outputs to {LIB_DIR}")


if __name__ == "__main__":
    main()