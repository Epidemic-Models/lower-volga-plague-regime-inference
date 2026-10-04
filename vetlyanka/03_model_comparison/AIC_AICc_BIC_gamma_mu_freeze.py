"""
AIC_AICc_BIC_gamma_mu_freeze.py   (step 03 -- run after all five fits in 02_fits)

Model comparison across the FIVE Vetlyanka models:

  Model            transmission     feedback   gamma, mu      k   T_perceive
  ---------------  ---------------  ---------  -------------  --  ------------------------------
  Null             single sigmoid   OFF (=0)   time-varying    9  irrelevant (feedback off)
  Single           single sigmoid   on         time-varying   10  FIXED, borrowed from Double
  Double           double sigmoid   on         time-varying   13  fitted
  Double-frozen    double sigmoid   on         constant       11  fitted
  Single-frozen    single sigmoid   on         constant        8  FIXED, borrowed from Double-frozen

Everything is READ from the fit outputs in 02_fits/results/ -- including the
fixed T_perceive values (stored as "fixed" rows in the single and
single-frozen fit CSVs), so nothing has to be typed in by hand.

Each model is re-simulated from its saved best-fit parameters, and SSE,
RMSE, AIC, AICc and BIC are computed on two bases:
  * cumulative deaths (the basis the fits were optimised on), and
  * incremental (weekly) deaths, as a secondary check.
As a consistency check, the re-computed cumulative SSE is compared with the
SSE written by each fit script.

OBSERVATION TIMES: weeks 1..22 compared at t = 1..22 (same as the fits).
LEGACY_ALIGNMENT = True reproduces the published t = 0..21 -- only
meaningful if the fits were also run in legacy mode.

Writes to 03_model_comparison/results/:
  vetlyanka_model_comparison_with_frozen_single.csv              (cumulative)
  vetlyanka_model_comparison_incremental_with_frozen_single.csv  (incremental)
  vetlyanka_pairwise_deltas.csv                                  (both bases)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # .../vetlyanka/03_model_comparison
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):  # walk up to vetlyanka/, which contains src/
    if os.path.dirname(ROOT) == ROOT:
        raise FileNotFoundError("Could not find the src/ folder above this script.")
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from plague_double_sigmoid_model import plague_model as plague_model_double
from plague_single_sigmoid_model import plague_model as plague_model_single
from vetlyanka_bounds import PARAM_NAMES

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
LEGACY_ALIGNMENT = False   # must match the setting used in the 02_fits scripts

FITS_DIR = os.path.join(ROOT, "02_fits", "results")
OUTPUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

t_start, t_end, dt = 0.0, 22.0, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
n_weeks = 22
OBS = slice(0, n_weeks) if LEGACY_ALIGNMENT else slice(1, n_weeks + 1)
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]  # S0, I0, R0, DI0, DDI0, DR0, P0

X1_C1_PLACEHOLDER = 1.0        # x1, c1 are not used by the single-sigmoid model
T_PERCEIVE_PLACEHOLDER = 1.0   # null model only: inert when sensitivity = 0

observed_cumulative = np.array([3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19, 27, 34,
                                90, 259, 313, 345, 364, 376, 376, 376, 376], dtype=float)

# label, model function, fit CSV, summary file, k
MODELS = [
    ("Null (no feedback)", "single",
     ("null", "vetlyanka_polished_fit_null.csv"), ("null", "vetlyanka_null_summary.txt"), 9),
    ("Single-sigmoid (+feedback)", "single",
     ("single", "vetlyanka_polished_fit_single.csv"), ("single", "vetlyanka_single_summary.txt"), 10),
    ("Double-sigmoid", "double",
     ("double", "vetlyanka_polished_fit.csv"), ("double", "vetlyanka_double_summary.txt"), 13),
    ("Double, frozen γ, μ", "double",
     ("double_frozen", "vetlyanka_constant_mu_gamma_polished_fit.csv"),
     ("double_frozen", "vetlyanka_constant_mu_gamma_summary.txt"), 11),
    ("Single, frozen γ, μ", "single",
     ("single_frozen", "vetlyanka_constant_mu_gamma_single_FIXED_fit.csv"),
     ("single_frozen", "vetlyanka_constant_mu_gamma_single_summary.txt"), 8),
]
SHORT = {"Null (no feedback)": "Null", "Single-sigmoid (+feedback)": "Single",
         "Double-sigmoid": "Double", "Double, frozen γ, μ": "Double-frozen",
         "Single, frozen γ, μ": "Single-frozen"}

# Pairs to report (first minus second; positive favours the second)
PAIRS = [("Single", "Double"), ("Single-frozen", "Double-frozen"),
         ("Null", "Single"), ("Null", "Double"),
         ("Double", "Double-frozen"), ("Single", "Single-frozen"),
         ("Single", "Double-frozen"), ("Double", "Single-frozen"),
         ("Null", "Double-frozen"), ("Null", "Single-frozen")]


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load_fit_dict(subdir, filename):
    path = os.path.join(FITS_DIR, subdir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run the matching script in 02_fits first.")
    df = pd.read_csv(path)
    return dict(zip(df["Parameter"], df["Best_Fit"].astype(float)))


def to_full_vector(fit):
    """Map a fit's saved parameters (any of the five models) onto the 13 model slots."""
    d = dict(fit)
    if "gamma_const" in d:   # frozen models
        d["gamma1"] = d["gamma2"] = d["gamma_const"]
        d["mu1"] = d["mu2"] = d["mu_const"]
    if "T_perceive" not in d:   # null model only (sensitivity = 0 makes it inert)
        if d.get("sensitivity", None) != 0.0:
            raise ValueError("T_perceive missing from a model with feedback switched on.")
        d["T_perceive"] = T_PERCEIVE_PLACEHOLDER
    missing = [n for n in PARAM_NAMES if n not in d and n not in ("x1", "c1")]
    if missing:
        raise ValueError(f"Fit file is missing parameters: {missing}")
    return np.array([d.get(name, X1_C1_PLACEHOLDER) for name in PARAM_NAMES], dtype=float)


def read_summary_sse(subdir, filename):
    path = os.path.join(FITS_DIR, subdir, filename)
    if not os.path.exists(path):
        return np.nan
    for line in open(path, encoding="utf-8"):
        if line.startswith("SSE="):
            return float(line.split("=", 1)[1])
    return np.nan


# ---------------------------------------------------------------------
# Simulation and criteria
# ---------------------------------------------------------------------
def simulate_weekly_cumulative(model_fn, full_params):
    sol = solve_ivp(model_fn, [t_start, t_end], initial_conditions,
                    args=(full_params,), t_eval=t_points, method="RK45")
    if not sol.success:
        raise RuntimeError("ODE solve failed for saved best-fit parameters -- check the fit CSV.")
    weekly = sol.y[5][::steps_per_week][OBS]   # DR = cumulative deaths at the observation times
    if len(weekly) != n_weeks:
        raise RuntimeError(f"Got {len(weekly)} weekly points, expected {n_weeks}.")
    return weekly


def information_criteria(observed, predicted, k):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = observed.size
    sse = float(np.sum((observed - predicted) ** 2))
    sse = max(sse, np.finfo(float).tiny)
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    aicc = np.inf if n <= k + 1 else aic + 2 * k * (k + 1) / (n - k - 1)
    return {"n": n, "k": k, "SSE": sse, "RMSE": float(np.sqrt(sse / n)),
            "AIC": float(aic), "AICc": float(aicc), "BIC": float(bic)}


def to_incremental(cumulative, baseline=0.0):
    """Weekly deaths. DR(0) = 0, so week 1's deaths = cumulative at week 1."""
    return np.diff(np.concatenate(([baseline], np.asarray(cumulative, dtype=float))))


def pairwise_table(stats, basis):
    rows = []
    for a, b in PAIRS:
        rows.append({"basis": basis, "comparison": f"{a} minus {b}",
                     "dAIC": stats[a]["AIC"] - stats[b]["AIC"],
                     "dAICc": stats[a]["AICc"] - stats[b]["AICc"],
                     "dBIC": stats[a]["BIC"] - stats[b]["BIC"],
                     "note": f"positive favours {b}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print(f"Observation times: {'t = 0..21 (LEGACY)' if LEGACY_ALIGNMENT else 't = 1..22 (end of each week)'}")
    print(f"Reading fits from: {FITS_DIR}\n")

    obs_incr = to_incremental(observed_cumulative)
    cum_stats, inc_stats, rows_cum, rows_inc = {}, {}, [], []
    for label, kind, (sub, csv), (ssub, summ), k in MODELS:
        fit = load_fit_dict(sub, csv)
        full = to_full_vector(fit)
        model_fn = plague_model_double if kind == "double" else plague_model_single
        pred = simulate_weekly_cumulative(model_fn, full)

        s_cum = information_criteria(observed_cumulative, pred, k)
        s_inc = information_criteria(obs_incr, to_incremental(pred), k)
        cum_stats[SHORT[label]], inc_stats[SHORT[label]] = s_cum, s_inc
        rows_cum.append({"Model": label, **s_cum, "T_perceive": full[PARAM_NAMES.index("T_perceive")]})
        rows_inc.append({"Model": label, **s_inc})

        fit_sse = read_summary_sse(ssub, summ)
        flag = "" if np.isnan(fit_sse) or abs(s_cum["SSE"] - fit_sse) <= 1e-5 * max(1.0, fit_sse) \
            else "   <-- DIFFERS from the fit script's SSE (check LEGACY_ALIGNMENT / files)"
        print(f"  {label:28s} k={k:2d}  SSE={s_cum['SSE']:.6f}  (fit script: {fit_sse:.6f}){flag}")

    results = pd.DataFrame(rows_cum).set_index("Model")
    results_incr = pd.DataFrame(rows_inc).set_index("Model")
    deltas = pd.concat([pairwise_table(cum_stats, "cumulative"),
                        pairwise_table(inc_stats, "incremental")], ignore_index=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\nModel comparison (cumulative basis -- the basis the fits were optimised on)")
    print(results.round(3))
    print("\nModel comparison (incremental / weekly basis)")
    print(results_incr.round(3))
    print("\nPairwise differences (first minus second; positive favours the second)")
    print(deltas.round(3).to_string(index=False))

    for basis, stats in (("cumulative", cum_stats), ("incremental", inc_stats)):
        print(f"\nBest model by criterion ({basis}):")
        for crit in ("AIC", "AICc", "BIC"):
            best = min(stats, key=lambda m: stats[m][crit])
            print(f"  {crit:5s}: {best} ({stats[best][crit]:.3f})")

    results.to_csv(os.path.join(OUTPUT_DIR, "vetlyanka_model_comparison_with_frozen_single.csv"))
    results_incr.to_csv(os.path.join(OUTPUT_DIR, "vetlyanka_model_comparison_incremental_with_frozen_single.csv"))
    deltas.to_csv(os.path.join(OUTPUT_DIR, "vetlyanka_pairwise_deltas.csv"), index=False)
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()