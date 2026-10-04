"""
bootstrap_frozen.py   (step 04 -- run after build_frozen_library.py and 03_model_comparison)

Parametric bootstrap (simulated-null) test for the CONSTANT-gamma/mu pair:
double-frozen (k = 11) vs single-frozen (k = 8). Same method as
parametric_bootstrap_test.py (sigmoidal pair) -- see that script for the full
explanation.

QUESTION: if the true process had only ONE regime (the fitted single-frozen
model), how often would the fitting procedure still favour double-frozen by as
much as it does on the real data?

PROCEDURE, per replicate:
  1. Synthetic data: Poisson noise on the weekly increments of the fitted
     single-frozen model (ground truth), cumulated.
  2. Starting points: best-matching candidate in each model's frozen library.
  3. Polish: one L-BFGS-B run (POLISH_MAXITER iterations) for BOTH models,
     within the real fits' bounds and constraints (single-frozen: T_perceive
     fixed; double-frozen: x1 - x0 >= 2, x1 <= 20, T_perceive free).
  4. Record delta = single-frozen - double-frozen for AIC, AICc, BIC
     (cumulative and incremental bases).
p-value = (1 + #{null delta >= observed delta}) / (1 + N).

LIBRARY: candidates are 11-dim reduced double-frozen vectors
(b1, b2, x0, x1, c, c1, gamma_const, mu_const, T_perceive, sensitivity, dispose_rate).
The double-frozen library = main rows + widened-x1 rows (build_frozen_library.py
saves the parameters for both); widened starts are clipped into the real bounds
before polishing. The single-frozen library = main rows only.

EVERYTHING IS READ, NOT TYPED IN:
  * observed deltas  <- 03_model_comparison/results/vetlyanka_pairwise_deltas.csv
                        ("Single-frozen minus Double-frozen")
  * ground truth      <- 02_fits/results/single_frozen/vetlyanka_constant_mu_gamma_single_FIXED_fit.csv
                         (including its fixed T_perceive)
  * library settings  <- 04_bootstrap/results/frozen/library_info.json
The script stops if the library's T_perceive or alignment does not match the fits.

Writes to 04_bootstrap/results/frozen/:
    bootstrap_null_distribution_frozen.csv
    bootstrap_pvalues_frozen.csv
    bootstrap_null_distribution_frozen.png
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))  # .../vetlyanka/04_bootstrap
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
from vetlyanka_bounds import PARAM_NAMES, BOUNDS

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
LEGACY_ALIGNMENT = False  # must match the fits and the library
N_BOOTSTRAP = 500
RNG_SEED = 0
POLISH_MAXITER = 200  # keep the same value as parametric_bootstrap_test.py
X0_X1_MIN_GAP = 2.0  # must match polish_vetlyanka_fit_double_freeze_mu_gamma_FIXED.py
K_SINGLE, K_DOUBLE = 8, 11

LIB_DIR = os.path.join(HERE, "results", "frozen")
SINGLE_FIT_PATH = os.path.join(ROOT, "02_fits", "results", "single_frozen",
                               "vetlyanka_constant_mu_gamma_single_FIXED_fit.csv")
DELTAS_PATH = os.path.join(ROOT, "03_model_comparison", "results", "vetlyanka_pairwise_deltas.csv")
OBSERVED_COMPARISON = "Single-frozen minus Double-frozen"

t_start, t_end, dt = 0.0, 22.0, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
n_weeks = 22
OBS = slice(0, n_weeks) if LEGACY_ALIGNMENT else slice(1, n_weeks + 1)
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]
X1_C1_PLACEHOLDER = 1.0

# Parameter structure
_B = dict(zip(PARAM_NAMES, BOUNDS))
GAMMA_CONST_BOUND = (min(_B["gamma1"][0], _B["gamma2"][0]), max(_B["gamma1"][1], _B["gamma2"][1]))
MU_CONST_BOUND = (min(_B["mu1"][0], _B["mu2"][0]), max(_B["mu1"][1], _B["mu2"][1]))
DF_NAMES = ["b1", "b2", "x0", "x1", "c", "c1", "gamma_const", "mu_const",
            "T_perceive", "sensitivity", "dispose_rate"]
DF_BOUNDS = [_B["b1"], _B["b2"], _B["x0"], _B["x1"], _B["c"], _B["c1"], GAMMA_CONST_BOUND,
             MU_CONST_BOUND, _B["T_perceive"], _B["sensitivity"], _B["dispose_rate"]]
SF_NAMES = ["b1", "b2", "x0", "c", "gamma_const", "mu_const", "sensitivity", "dispose_rate"]
SF_BOUNDS = [_B["b1"], _B["b2"], _B["x0"], _B["c"], GAMMA_CONST_BOUND, MU_CONST_BOUND,
             _B["sensitivity"], _B["dispose_rate"]]
DF_TO_SF_IDX = [DF_NAMES.index(n) for n in SF_NAMES]  # single-frozen values inside an 11-dim library row
IDX_X0, IDX_X1 = DF_NAMES.index("x0"), DF_NAMES.index("x1")
LO_D = np.array([b[0] for b in DF_BOUNDS], float);
HI_D = np.array([b[1] for b in DF_BOUNDS], float)
LO_S = np.array([b[0] for b in SF_BOUNDS], float);
HI_S = np.array([b[1] for b in SF_BOUNDS], float)


def double_frozen_full(r):
    b1, b2, x0, x1, c, c1, g, m, tp, sens, disp = r
    return np.array([b1, b2, x0, x1, c, c1, g, g, m, m, tp, sens, disp], dtype=float)


def make_single_full(t_perceive):
    def single_full(r):
        d = dict(zip(SF_NAMES, r))
        d["gamma1"] = d["gamma2"] = d["gamma_const"]
        d["mu1"] = d["mu2"] = d["mu_const"]
        d["T_perceive"] = t_perceive
        return np.array([d.get(n, X1_C1_PLACEHOLDER) for n in PARAM_NAMES], dtype=float)

    return single_full


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------
def require(path, hint):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- {hint}")
    return path


def load_inputs():
    with open(require(os.path.join(LIB_DIR, "library_info.json"), "run build_frozen_library.py first.")) as fh:
        info = json.load(fh)
    if info.get("legacy_alignment") != LEGACY_ALIGNMENT:
        raise ValueError("Library alignment does not match LEGACY_ALIGNMENT -- rebuild the library.")
    if info.get("reduced_names") != DF_NAMES:
        raise ValueError("Library parameter order differs from this script -- rebuild the library.")

    fit = pd.read_csv(require(SINGLE_FIT_PATH,
                              "run 02_fits/polish_vetlyanka_fit_single_freeze_mu_gamma_FIXED.py first."))
    fit = dict(zip(fit["Parameter"], fit["Best_Fit"].astype(float)))
    t_perceive = fit["T_perceive"]
    if abs(t_perceive - info["T_perceive_fixed_single"]) > 1e-9:
        raise ValueError(f"Library was built with T_perceive={info['T_perceive_fixed_single']} but the "
                         f"single-frozen fit uses {t_perceive} -- re-run build_frozen_library.py.")

    double_traj = np.load(os.path.join(LIB_DIR, "double_frozen_trajectories.npy"))
    single_traj = np.load(os.path.join(LIB_DIR, "single_frozen_trajectories.npy"))
    params_main = np.load(os.path.join(LIB_DIR, "candidate_params_frozen_main.npy"))
    params_wide = np.load(require(os.path.join(LIB_DIR, "candidate_params_frozen_widened.npy"),
                                  "re-run build_frozen_library.py (the new version saves these)."))
    double_params = np.vstack([params_main, params_wide])
    if double_params.shape[0] != double_traj.shape[0] or params_main.shape[0] != single_traj.shape[0]:
        raise ValueError("Library rows and parameter rows do not line up -- rebuild the library.")

    deltas = pd.read_csv(require(DELTAS_PATH, "run 03_model_comparison/AIC_AICc_BIC_gamma_mu_freeze.py first."))
    observed = {}
    for basis, tag in (("cumulative", "cum"), ("incremental", "incr")):
        row = deltas[(deltas["basis"] == basis) & (deltas["comparison"] == OBSERVED_COMPARISON)].iloc[0]
        for crit in ("AIC", "AICc", "BIC"):
            observed[f"delta_{crit}_{tag}"] = float(row[f"d{crit}"])
    return fit, t_perceive, double_traj, single_traj, double_params, params_main, observed


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


def sse(model_fn, full, target):
    pred = simulate_weekly(model_fn, full)
    if pred is None:
        return 1e12
    v = float(np.sum((target - pred) ** 2))
    return v if np.isfinite(v) else 1e12


def sse_double(reduced, target):
    if reduced[IDX_X0] >= reduced[IDX_X1] - X0_X1_MIN_GAP:
        return 1e12
    return sse(plague_model_double, double_frozen_full(reduced), target)


def polish(objective, start, lo, hi):
    start = np.clip(start, lo, hi)
    sf = np.maximum(np.abs(start), 1e-3)
    res = minimize(lambda p: objective(p * sf), start / sf, method="L-BFGS-B",
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
    fit, t_perceive, double_traj, single_traj, double_params, single_params, observed = load_inputs()
    single_full = make_single_full(t_perceive)

    dv = ~np.isnan(double_traj).any(axis=1);
    sv = ~np.isnan(single_traj).any(axis=1)
    double_traj_v, double_params_v = double_traj[dv], double_params[dv]
    single_traj_v, single_params_v = single_traj[sv], single_params[sv]

    true_cumulative = simulate_weekly(plague_model_single, single_full(np.array([fit[n] for n in SF_NAMES])))
    true_increments = np.clip(to_incremental(true_cumulative), 0, None)

    print(f"Observation times: {'t = 0..21 (LEGACY)' if LEGACY_ALIGNMENT else 't = 1..22 (end of each week)'}")
    print(f"Library: {double_traj_v.shape[0]} valid double-frozen candidates "
          f"({double_traj.shape[0]} rows incl. widened-x1), {single_traj_v.shape[0]} valid single-frozen candidates")
    print(f"Ground truth: single-frozen fit, T_perceive fixed at {t_perceive:.6f}")
    print(f"Observed deltas ({OBSERVED_COMPARISON}, from 03_model_comparison): "
          + ", ".join(f"{k}={v:.3f}" for k, v in observed.items()))
    print(f"Running {N_BOOTSTRAP} replicates (seed {RNG_SEED}, polish maxiter {POLISH_MAXITER})...\n")

    rng = np.random.default_rng(RNG_SEED)
    records, n_failed, t0 = [], 0, time.time()
    for b in range(N_BOOTSTRAP):
        synth_cum = np.cumsum(rng.poisson(true_increments).astype(float))

        start_d = double_params_v[np.argmin(np.sum((double_traj_v - synth_cum) ** 2, axis=1))]
        start_s = single_params_v[np.argmin(np.sum((single_traj_v - synth_cum) ** 2, axis=1))]

        pol_d = double_frozen_full(polish(lambda p: sse_double(p, synth_cum), start_d, LO_D, HI_D))
        pol_s = single_full(polish(lambda p: sse(plague_model_single, single_full(p), synth_cum),
                                   start_s[DF_TO_SF_IDX], LO_S, HI_S))
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
                        "SSE_double": float(np.sum((synth_cum - best_d) ** 2)),
                        "SSE_single": float(np.sum((synth_cum - best_s) ** 2)),
                        "delta_AIC_cum": aic_s - aic_d, "delta_AICc_cum": aicc_s - aicc_d,
                        "delta_BIC_cum": bic_s - bic_d,
                        "delta_AIC_incr": aic_si - aic_di, "delta_AICc_incr": aicc_si - aicc_di,
                        "delta_BIC_incr": bic_si - bic_di})
        if (b + 1) % 25 == 0:
            rate = (b + 1) / (time.time() - t0)
            print(f"  replicate {b + 1}/{N_BOOTSTRAP}  (~{(N_BOOTSTRAP - b - 1) / rate / 60:.0f} min left)", flush=True)

    null_df = pd.DataFrame(records)
    null_df.to_csv(os.path.join(LIB_DIR, "bootstrap_null_distribution_frozen.csv"), index=False)
    n = len(null_df)
    print(f"\nCompleted replicates: {n} (failed solves skipped: {n_failed})")
    print(f"Mean SSE in replicates -- double-frozen: {null_df['SSE_double'].mean():.1f}, "
          f"single-frozen: {null_df['SSE_single'].mean():.1f}  "
          f"(double-frozen worse in {int((null_df['SSE_double'] > null_df['SSE_single']).sum())}/{n})")

    rows = []
    for col, obs_val in observed.items():
        vals = null_df[col].values
        n_ge = int(np.sum(vals >= obs_val))
        rows.append({"statistic": col, "observed": obs_val, "null_mean": vals.mean(), "null_sd": vals.std(ddof=1),
                     "null_95th_pct": np.percentile(vals, 95), "null_max": vals.max(),
                     "n_null_ge_observed": n_ge, "n_replicates": n,
                     "p_value": (1 + n_ge) / (1 + n), "p_value_plain_fraction": n_ge / n})
    pvals = pd.DataFrame(rows)
    pvals.to_csv(os.path.join(LIB_DIR, "bootstrap_pvalues_frozen.csv"), index=False)
    pd.set_option("display.width", 200);
    pd.set_option("display.max_columns", None)
    print("\nObserved delta vs null distribution (positive delta favours double-frozen):")
    print(pvals.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_df["delta_AICc_cum"], bins=25, color="steelblue", edgecolor="white")
    ax.axvline(observed["delta_AICc_cum"], color="crimson", linewidth=2, label="Observed (real data)")
    ax.set_xlabel("ΔAICc (single-frozen − double-frozen), cumulative basis")
    ax.set_ylabel("Bootstrap replicates")
    ax.set_title("Null distribution: data simulated from the single-frozen fit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(LIB_DIR, "bootstrap_null_distribution_frozen.png"), dpi=200)
    print(f"\nSaved outputs to {LIB_DIR}")


if __name__ == "__main__":
    main()

