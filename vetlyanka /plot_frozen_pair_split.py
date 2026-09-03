"""
plot_frozen_pair_split.py

One figure, two panels side by side: left is cumulative deaths, right is
weekly deaths -- both models (double-frozen and single-frozen) overlaid
together on each panel, distinguished by color, against the same
observed data points.

T_perceive: single-frozen's fit fixes T_perceive at 3.409 rather than
fitting it (frozen-double's own converged value, the correct family
match -- see polish_vetlyanka_fit_single_freeze_mu_gamma_FIXED.py) --
not a row in the current fit CSV, supplied here as the same fixed
constant. Double-frozen's T_perceive remains genuinely fitted and is
read from its own CSV.

DATA LAYOUT: reads both fits from this project's data/fits/ folder.

Output (in figures/):
    Fig_frozen_pair_grid.pdf / .png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from plague_double_sigmoid_model import plague_model as plague_model_double
from plague_single_sigmoid_model import plague_model as plague_model_single

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

T_PERCEIVE_FIXED_SINGLE = 3.409  # must match polish_vetlyanka_fit_single_freeze_mu_gamma_FIXED.py

initial_conditions = [1699, 1, 0, 0, 0, 0, 0]
newinfectedweekly = np.array([3, 0, 2, 0, 1, 0, 2, 3, 0, 1, 7, 8, 7, 56, 169,
                               54, 32, 19, 12, 0, 0, 0])
cumulativeweekly_all_regions = np.cumsum(newinfectedweekly)
weekly_time = np.arange(0, 22)
dt = 0.01
t_start, t_end = 0, 22
time_points = np.arange(t_start, t_end + dt, dt)
steps_by_weeks = int(1 / dt)


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


double_frozen_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "fits", "double_frozen", "vetlyanka_constant_mu_gamma_polished_fit.csv")
).set_index("Parameter")
order = ["b1", "b2", "x0", "x1", "c", "c1", "gamma_const", "mu_const",
         "T_perceive", "sensitivity", "dispose_rate"]
double_frozen_reduced = np.array([double_frozen_df.loc[n, "Best_Fit"] for n in order])
double_frozen_full = expand_double_frozen_to_full(double_frozen_reduced)

single_frozen_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "fits", "single_frozen", "vetlyanka_constant_mu_gamma_single_FIXED_fit.csv")
)
single_frozen_full = expand_single_frozen_to_full(single_frozen_df)

sol_double = solve_ivp(plague_model_double, [t_start, t_end], initial_conditions,
                        args=(double_frozen_full,), t_eval=time_points, method="RK45")
sol_single = solve_ivp(plague_model_single, [t_start, t_end], initial_conditions,
                        args=(single_frozen_full,), t_eval=time_points, method="RK45")

DR_double = sol_double.y[5]
DR_single = sol_single.y[5]
# Smooth, continuous instantaneous death rate (deaths/week), rather than
# 22 discretized weekly sums connected by straight lines -- np.gradient on
# the fine-resolution cumulative-death trajectory gives the genuine, smooth
# shape the ODE actually has, converging to the same weekly totals when
# integrated over a week.
double_rate_smooth = np.gradient(DR_double, time_points)
single_rate_smooth = np.gradient(DR_single, time_points)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.8,
    "lines.markersize": 4,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
})

COLOR_DOUBLE = "#B3222B"
COLOR_SINGLE = "#E08214"
COLOR_OBS = "black"

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), dpi=300)

ax_cum, ax_weekly = axes[0], axes[1]

# Cumulative panel -- both models overlaid
ax_cum.plot(time_points, DR_double, color=COLOR_DOUBLE, linestyle="-", label="Double, constant γ,μ")
ax_cum.plot(time_points, DR_single, color=COLOR_SINGLE, linestyle="-", label="Single, constant γ,μ")
ax_cum.scatter(weekly_time, cumulativeweekly_all_regions, color=COLOR_OBS, marker="o",
               s=14, zorder=5, linewidths=0, label="Observed")
ax_cum.set_title("Cumulative deaths", fontsize=9, fontweight="bold")
ax_cum.set_xlabel("Time (weeks)")
ax_cum.set_ylabel("Cumulative deaths")
ax_cum.legend(frameon=False, loc="upper left", handlelength=1.6, fontsize=7)
ax_cum.spines[["top", "right"]].set_visible(False)
ax_cum.grid(True, linewidth=0.4, alpha=0.35)

# Weekly (incremental) panel -- both models overlaid
ax_weekly.plot(time_points[50:], double_rate_smooth[50:], color=COLOR_DOUBLE, linestyle="-", label="Double, constant γ,μ")
ax_weekly.plot(time_points[50:], single_rate_smooth[50:], color=COLOR_SINGLE, linestyle="-", label="Single, constant γ,μ")
ax_weekly.scatter(weekly_time, newinfectedweekly, color=COLOR_OBS, marker="o",
                   s=14, zorder=5, linewidths=0, label="Observed")
ax_weekly.set_title("Weekly deaths", fontsize=9, fontweight="bold")
ax_weekly.set_xlabel("Time (weeks)")
ax_weekly.set_ylabel("Weekly deaths")
ax_weekly.legend(frameon=False, loc="upper left", handlelength=1.6, fontsize=7)
ax_weekly.spines[["top", "right"]].set_visible(False)
ax_weekly.grid(True, linewidth=0.4, alpha=0.35)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Fig_frozen_pair_grid.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "Fig_frozen_pair_grid.png"), dpi=600, bbox_inches="tight")
print(f"Saved {FIG_DIR}/Fig_frozen_pair_grid.pdf and .png")
plt.show()
plt.close(fig)