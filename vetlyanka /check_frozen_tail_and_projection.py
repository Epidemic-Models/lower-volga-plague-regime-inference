"""
check_frozen_tail_and_projection.py

Mechanistic check for the FROZEN gamma/mu models (double-frozen vs
single-frozen), mirroring the original sigmoidal-gamma/mu check:
  1. Weekly-new-deaths tail residuals (weeks 19-21) against the real,
     complete halt (observed: 12, 0, 0, 0).
  2. Extended projection to week 52 -- does single-frozen predict a
     rebound wave the way the sigmoidal single-sigmoid model did?

T_perceive: single-frozen's fit fixes T_perceive at 3.409 (frozen-
double's own converged value, the correct family match -- see
polish_vetlyanka_fit_single_freeze_mu_gamma_FIXED.py) rather than
fitting it -- it is NOT a row in the current fit CSV, so it is supplied
here as the same fixed constant rather than read from the file.
Double-frozen's T_perceive remains genuinely fitted and is read from
its own CSV as before.

DATA LAYOUT: reads both fits from this project's data/fits/ folder.

Required files:
    vetlyanka_bounds.py, plague_double_sigmoid_model.py,
    plague_single_sigmoid_model.py,
    data/fits/double_frozen/vetlyanka_constant_mu_gamma_polished_fit.csv,
    data/fits/single_frozen/vetlyanka_constant_mu_gamma_single_FIXED_fit.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from plague_double_sigmoid_model import plague_model as plague_model_double
from plague_single_sigmoid_model import plague_model as plague_model_single
from vetlyanka_bounds import PARAM_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]
T_PERCEIVE_FIXED_SINGLE = 3.409  # must match polish_vetlyanka_fit_single_freeze_mu_gamma_FIXED.py
newinfectedweekly = np.array([3, 0, 2, 0, 1, 0, 2, 3, 0, 1, 7, 8, 7, 56, 169,
                               54, 32, 19, 12, 0, 0, 0])


def expand_double_frozen_to_full(freeze_params):
    (b1, b2, x0, x1, c, c1, gamma_const, mu_const,
     T_perceive, sensitivity, dispose_rate) = freeze_params
    return np.array([b1, b2, x0, x1, c, c1, gamma_const, gamma_const,
                      mu_const, mu_const, T_perceive, sensitivity, dispose_rate])


def expand_single_frozen_to_full(single_frozen_df):
    df = single_frozen_df.set_index("Parameter")
    b1, b2, x0, c = [df.loc[n, "Best_Fit"] for n in ["b1", "b2", "x0", "c"]]
    gamma_const, mu_const = [df.loc[n, "Best_Fit"] for n in ["gamma_const", "mu_const"]]
    sensitivity, dispose_rate = [df.loc[n, "Best_Fit"] for n in ["sensitivity", "dispose_rate"]]
    T_perceive = T_PERCEIVE_FIXED_SINGLE  # fixed, not read from file -- see module docstring
    return np.array([b1, b2, x0, 1.0, c, 1.0, gamma_const, gamma_const,
                      mu_const, mu_const, T_perceive, sensitivity, dispose_rate])


double_frozen_path = os.path.join(
    BASE_DIR, "data", "fits", "double_frozen", "vetlyanka_constant_mu_gamma_polished_fit.csv"
)
single_frozen_path = os.path.join(
    BASE_DIR, "data", "fits", "single_frozen", "vetlyanka_constant_mu_gamma_single_FIXED_fit.csv"
)

double_frozen_df = pd.read_csv(double_frozen_path).set_index("Parameter")
order = ["b1", "b2", "x0", "x1", "c", "c1", "gamma_const", "mu_const",
         "T_perceive", "sensitivity", "dispose_rate"]
double_frozen_reduced = np.array([double_frozen_df.loc[n, "Best_Fit"] for n in order])
double_frozen_full = expand_double_frozen_to_full(double_frozen_reduced)

single_frozen_df = pd.read_csv(single_frozen_path)
single_frozen_full = expand_single_frozen_to_full(single_frozen_df)

# ------------------------------------------------------------
# 1. Tail-residual check (weeks 19-21)
# ------------------------------------------------------------
t_points_22 = np.arange(0, 22.01, 0.01)
sol_d22 = solve_ivp(plague_model_double, [0, 22], initial_conditions,
                     args=(double_frozen_full,), t_eval=t_points_22, method="RK45")
sol_s22 = solve_ivp(plague_model_single, [0, 22], initial_conditions,
                     args=(single_frozen_full,), t_eval=t_points_22, method="RK45")

DR_d = sol_d22.y[5][::100][:22]
DR_s = sol_s22.y[5][::100][:22]
weekly_d = np.diff(np.concatenate(([0.0], DR_d)))
weekly_s = np.diff(np.concatenate(([0.0], DR_s)))

print("=== Tail residual check (weeks 19-21) ===")
print(f"Observed:      {newinfectedweekly[19:22]}")
print(f"Double-frozen: {np.round(weekly_d[19:22], 2)}")
print(f"Single-frozen: {np.round(weekly_s[19:22], 2)}")

# ------------------------------------------------------------
# 2. Extended projection to week 52
# ------------------------------------------------------------
t_points_52 = np.arange(0, 52.01, 0.01)
sol_d52 = solve_ivp(plague_model_double, [0, 52], initial_conditions,
                     args=(double_frozen_full,), t_eval=t_points_52, method="RK45")
sol_s52 = solve_ivp(plague_model_single, [0, 52], initial_conditions,
                     args=(single_frozen_full,), t_eval=t_points_52, method="RK45")

DR_d52 = sol_d52.y[5][::100][:52]
DR_s52 = sol_s52.y[5][::100][:52]
weekly_d52 = np.diff(np.concatenate(([0.0], DR_d52)))
weekly_s52 = np.diff(np.concatenate(([0.0], DR_s52)))

I_d = sol_d52.y[1]
I_s = sol_s52.y[1]
t = sol_d52.t
trough_mask = (t >= 18) & (t <= 25)

print("\n=== Extended projection (weeks 22-51) ===")
print(f"Double-frozen weekly deaths wk22/25/30/40: "
      f"{np.round([weekly_d52[w] for w in [22, 25, 30, 40]], 2)}")
print(f"Single-frozen weekly deaths wk22/25/30/40: "
      f"{np.round([weekly_s52[w] for w in [22, 25, 30, 40]], 2)}")
print(f"Double-frozen min I in trough (wk18-25): {I_d[trough_mask].min():.4f}")
print(f"Single-frozen min I in trough (wk18-25): {I_s[trough_mask].min():.4f}")