"""
check_rebound_not_numerical_artefact.py   (step 06 -- run after 02_fits)

Tests whether the single-regime models' predicted resurgence after the real
halt (weeks 20-22 onward) is a genuine feature of the fitted ODE system, or a
numerical-integration artefact (e.g. I(t) driven to a meaningless residual in
the trough, then regrowing from floating-point noise).

Covers BOTH single-regime models, each with its saved best fit:
  * Single        (time-varying gamma, mu) -- 02_fits/results/single/
  * Single-frozen (constant gamma, mu)     -- 02_fits/results/single_frozen/
(This replaces check_frozen_rebound_not_numerical_artefact.py.)

Two checks per model:
  1. Is I(t) at its minimum in the trough (weeks 18-25) a meaningful number
     (order 0.1-1 or more, comparable to the seed I0 = 1), or a near-zero
     numerical residue (e.g. 1e-10)?
  2. Do the predicted deaths change if the ODE is solved with 10,000x tighter
     tolerances, or with an implicit stiff solver (Radau)? An artefact would
     change or vanish; a genuine solution agrees across all three.

Deaths are reported both all-cause (DR, includes ~0.85/week background
mortality) and plague-only (DDI). Week k = deaths between t = k-1 and t = k.

Writes: 06_robustness/results/rebound_check.csv
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # .../vetlyanka/06_robustness
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):  # walk up to vetlyanka/, which contains src/
    if os.path.dirname(ROOT) == ROOT:
        raise FileNotFoundError("Could not find the src/ folder above this script.")
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from plague_single_sigmoid_model import plague_model
from vetlyanka_bounds import PARAM_NAMES

FITS_DIR = os.path.join(ROOT, "02_fits", "results")
OUTPUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    ("Single (time-varying gamma, mu)", os.path.join(FITS_DIR, "single", "vetlyanka_polished_fit_single.csv")),
    ("Single-frozen (constant gamma, mu)",
     os.path.join(FITS_DIR, "single_frozen", "vetlyanka_constant_mu_gamma_single_FIXED_fit.csv")),
]
RUNS = [
    ("RK45, default tolerances", dict(method="RK45")),                       # rtol=1e-3, atol=1e-6
    ("RK45, tolerances 10,000x tighter", dict(method="RK45", rtol=1e-10, atol=1e-12)),
    ("Radau (implicit), tight tolerances", dict(method="Radau", rtol=1e-10, atol=1e-12)),
]
REPORT_WEEKS = [20, 21, 22, 25, 30]
X1_C1_PLACEHOLDER = 1.0
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]  # S0, I0, R0, DI0, DDI0, DR0, P0
t_end = 52
t_eval = np.arange(0, t_end + 0.005, 0.01)


def load_full_params(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run the matching 02_fits script first.")
    df = pd.read_csv(path)
    d = dict(zip(df["Parameter"], df["Best_Fit"].astype(float)))
    if "gamma_const" in d:
        d["gamma1"] = d["gamma2"] = d["gamma_const"]
        d["mu1"] = d["mu2"] = d["mu_const"]
    if "T_perceive" not in d:
        raise ValueError(f"No T_perceive row in {path} -- re-run the fit with the new 02_fits script.")
    return np.array([d.get(n, X1_C1_PLACEHOLDER) for n in PARAM_NAMES], dtype=float)


def main():
    rows = []
    for model_label, path in MODELS:
        params = load_full_params(path)
        print(f"\n{model_label}   (T_perceive = {params[PARAM_NAMES.index('T_perceive')]:.6f})")
        print(f"  {'Solver':<38} {'min I, wk18-25':>15} " +
              " ".join(f"{'wk' + str(w):>7}" for w in REPORT_WEEKS) + f" {'plague wk23-52':>15}")
        for run_label, kw in RUNS:
            sol = solve_ivp(plague_model, [0, t_end], initial_conditions, args=(params,), t_eval=t_eval, **kw)
            if not sol.success:
                print(f"  {run_label:<38} solver failed: {sol.message}")
                continue
            t, I = sol.t, sol.y[1]
            weekly_idx = np.searchsorted(t, np.arange(0, t_end + 1) - 1e-9)
            DR, DDI = sol.y[5][weekly_idx], sol.y[4][weekly_idx]
            all_cause, plague = np.diff(DR), np.diff(DDI)   # index k-1 = week k
            i_min = I[(t >= 18) & (t <= 25)].min()
            plague_later = plague[22:52].sum()                # weeks 23..52
            print(f"  {run_label:<38} {i_min:>15.6f} " +
                  " ".join(f"{plague[w - 1]:>7.3f}" for w in REPORT_WEEKS) + f" {plague_later:>15.1f}")
            row = {"model": model_label, "solver": run_label, "min_I_wk18_25": i_min,
                   "plague_deaths_wk23_52": plague_later}
            for w in REPORT_WEEKS:
                row[f"plague_wk{w}"] = plague[w - 1]
                row[f"allcause_wk{w}"] = all_cause[w - 1]
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "rebound_check.csv"), index=False)

    print("\nAgreement across solvers (max relative spread):")
    for model_label, g in df.groupby("model", sort=False):
        cols = [c for c in g.columns if c.startswith("plague_") or c == "min_I_wk18_25"]
        spread = max((g[c].max() - g[c].min()) / max(abs(g[c]).max(), 1e-12) for c in cols)
        print(f"  {model_label:<38} {100 * spread:.4f}%")
    print("\nWeekly columns are PLAGUE-only deaths (DDI); all-cause values (adding ~0.85/week background")
    print("mortality) are in the CSV. Observed data: 0 deaths in weeks 20-22.")
    print("Interpretation: rows that agree closely = genuine ODE solution, not a numerical artefact;")
    print("min I of order 0.1-1 or larger = real ongoing transmission, not a numerical residue.")
    print(f"\nSaved {os.path.join(OUTPUT_DIR, 'rebound_check.csv')}")


if __name__ == "__main__":
    main()