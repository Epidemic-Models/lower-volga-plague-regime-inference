"""
check_frozen_rebound_not_numerical_artefact.py

Confirms the single-frozen model's projected rebound wave is a genuine
property of the fitted ODE system, not a numerical-integration artifact,
by comparing results across default and 10,000x-tightened RK45
tolerances, plus an entirely different (implicit, stiff-solver)
integration algorithm, Radau.

T_perceive: single-frozen's fit fixes T_perceive at 3.409 (frozen-
double's own converged value, the correct family match) -- not a row
in the fit CSV, supplied here as the same fixed constant used in the
real fitting script.

DATA LAYOUT: reads the fit from data/fits/single_frozen/.

Requires:
    data/fits/single_frozen/vetlyanka_constant_mu_gamma_single_FIXED_fit.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from plague_single_sigmoid_model import plague_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FITS_SINGLE_FROZEN_DIR = os.path.join(BASE_DIR, "data", "fits", "single_frozen")

initial_conditions = [1699, 1, 0, 0, 0, 0, 0]
t_start, t_end_extended = 0, 52
t_eval = np.arange(t_start, t_end_extended + 0.01, 0.01)

T_PERCEIVE_FIXED_SINGLE = 3.409  # must match polish_vetlyanka_fit_single_freeze_mu_gamma_FIXED.py

single_frozen_df = pd.read_csv(
    os.path.join(FITS_SINGLE_FROZEN_DIR, "vetlyanka_constant_mu_gamma_single_FIXED_fit.csv")
).set_index("Parameter")

b1, b2, x0, c = [single_frozen_df.loc[n, "Best_Fit"] for n in ["b1", "b2", "x0", "c"]]
gamma_const, mu_const = [single_frozen_df.loc[n, "Best_Fit"] for n in ["gamma_const", "mu_const"]]
sensitivity, dispose_rate = [single_frozen_df.loc[n, "Best_Fit"] for n in ["sensitivity", "dispose_rate"]]
T_perceive = T_PERCEIVE_FIXED_SINGLE  # fixed, not read from file -- see module docstring

params = np.array([b1, b2, x0, 1.0, c, 1.0, gamma_const, gamma_const,
                    mu_const, mu_const, T_perceive, sensitivity, dispose_rate])

runs = [
    ("Default RK45 (loose tolerances)", dict(method="RK45")),
    ("RK45, tolerances tightened 10,000x", dict(method="RK45", rtol=1e-10, atol=1e-12)),
    ("Radau (implicit stiff-solver), same tight tolerances", dict(method="Radau", rtol=1e-10, atol=1e-12)),
]

print(f"{'Method':<55} {'min I, wk18-25':>16} {'wk22':>8} {'wk25':>8} {'wk30':>8}")
for label, solver_kwargs in runs:
    sol = solve_ivp(plague_model, [t_start, t_end_extended], initial_conditions,
                     args=(params,), t_eval=t_eval, **solver_kwargs)

    I = sol.y[1]
    t = sol.t
    trough_mask = (t >= 18) & (t <= 25)
    I_trough_min = I[trough_mask].min()

    DR = sol.y[5]
    weekly_DR = DR[::100][:52]
    weekly_deaths = np.diff(np.concatenate(([0.0], weekly_DR)))

    print(f"{label:<55} {I_trough_min:>16.6f} {weekly_deaths[22]:>8.3f} "
          f"{weekly_deaths[25]:>8.3f} {weekly_deaths[30]:>8.3f}")

print("\nInterpretation:")
print("- If these three rows agree closely, the rebound is the genuine solution")
print("  of the fitted ODE system, not a numerical-precision artefact.")
print("- I in the trough should also be checked against 0: a value of order")
print("  0.1-1 (comparable to the model's own I0=1 starting seed) indicates a")
print("  real, if small, level of ongoing transmission -- not noise.")