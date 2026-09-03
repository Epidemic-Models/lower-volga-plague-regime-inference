"""
plot_frozen_tail_zoom.py

Zoomed-in view of the mortality TAIL and extended projection for the
frozen-gamma/mu pair -- the region invisible on the main 0-160 weekly-
deaths axis, but where the actual mechanistic finding lives:
double-frozen reproduces the real halt (weeks 19-21: observed 0,0,0);
single-frozen does not, predicting a persistent trickle that resolves
into a rebound wave.

Extends the simulation to week 52 as a diagnostic of each fitted
mechanism's own long-run behaviour (not a forecast), matching
check_frozen_tail_and_projection.py exactly. Real observed data only
exists through week 21; nothing is plotted as "observed" beyond that.

T_perceive: single-frozen's fit fixes T_perceive at 3.409 (frozen-
double's own converged value) -- see
polish_vetlyanka_fit_single_freeze_mu_gamma_FIXED.py.

DATA LAYOUT: reads both fits from this project's data/fits/ folder.

Output (in figures/):
    Fig_frozen_tail_zoom.pdf / .png
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
                               54, 32, 19, 12, 0, 0, 0])  # weeks 0-21, real data only
n_weeks_observed = len(newinfectedweekly)

dt = 0.01
t_start, t_end = 0, 52  # extended projection
time_points = np.arange(t_start, t_end + dt, dt)
weekly_time = np.arange(0, 52)
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
    T_perceive = T_PERCEIVE_FIXED_SINGLE  # fixed, not read from file
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

# Smooth, continuous instantaneous death rate (deaths/week), rather than
# discretized weekly sums connected by straight lines -- np.gradient on the
# fine-resolution cumulative-death trajectory gives the genuine, smooth
# shape the ODE actually has, converging to the same weekly totals when
# integrated over a week. This is the correct smooth analog of "weekly
# deaths" for a continuous model, not an artificial smoothing hack.
double_rate_smooth = np.gradient(sol_double.y[5], time_points)
single_rate_smooth = np.gradient(sol_single.y[5], time_points)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.8,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
})

COLOR_DOUBLE = "#B3222B"
COLOR_SINGLE = "#E08214"
COLOR_OBS = "black"

fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=300)

ax.plot(time_points[50:], double_rate_smooth[50:], color=COLOR_DOUBLE, linestyle="-", label="Double, constant γ,μ")
ax.plot(time_points[50:], single_rate_smooth[50:], color=COLOR_SINGLE, linestyle="-", label="Single, constant γ,μ")
ax.scatter(np.arange(n_weeks_observed), newinfectedweekly, color=COLOR_OBS, marker="o",
           s=14, zorder=5, linewidths=0, label="Observed")
ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=1)
ax.axvline(21, color="gray", linewidth=0.6, linestyle=":", zorder=1)
ax.text(21.3, ax.get_ylim()[1] * 0.92, "end of\nobservation", fontsize=6.5, color="gray")

ax.set_xlim(15, 52)
ax.set_title("Tail and extended projection, constant γ,μ pair", fontsize=9, fontweight="bold")
ax.set_xlabel("Time (weeks)")
ax.set_ylabel("Weekly deaths")
ax.legend(frameon=False, loc="upper right", handlelength=1.6, fontsize=7)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, linewidth=0.4, alpha=0.35)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Fig_frozen_tail_zoom.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "Fig_frozen_tail_zoom.png"), dpi=600, bbox_inches="tight")
print(f"Saved {FIG_DIR}/Fig_frozen_tail_zoom.pdf and .png")
plt.show()
plt.close(fig)