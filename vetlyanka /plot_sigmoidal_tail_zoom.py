"""
plot_sigmoidal_tail_zoom.py

Same treatment as plot_frozen_tail_zoom.py, applied to the ORIGINAL
sigmoidal-gamma/mu double and single models. Extends the simulation to
week 52 as a diagnostic of each fitted mechanism's own long-run
behaviour (not a forecast), matching check_rebound_not_numerical_artefact.py.
Real observed data only exists through week 21.

T_perceive: single-sigmoid's fit fixes T_perceive at 3.571 (double-
sigmoid's own converged value) -- see polish_vetlyanka_fit_single_FIXED.py.

DATA LAYOUT: reads both fits from this project's data/fits/ folder.

Output (in figures/):
    Fig_sigmoidal_tail_zoom.pdf / .png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from plague_double_sigmoid_model import plague_model as plague_model_double
from plague_single_sigmoid_model import plague_model as plague_model_single
from vetlyanka_bounds import PARAM_NAMES, PARAM_NAMES_SINGLE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

T_PERCEIVE_FIXED_SIGMOID = 3.571  # must match polish_vetlyanka_fit_single_FIXED.py
PARAM_NAMES_SINGLE_SIGMOID_FIT = [n for n in PARAM_NAMES_SINGLE if n != "T_perceive"]
X1_C1_PLACEHOLDER = 1.0

initial_conditions = [1699, 1, 0, 0, 0, 0, 0]
newinfectedweekly = np.array([3, 0, 2, 0, 1, 0, 2, 3, 0, 1, 7, 8, 7, 56, 169,
                               54, 32, 19, 12, 0, 0, 0])  # weeks 0-21, real data only
n_weeks_observed = len(newinfectedweekly)

dt = 0.01
t_start, t_end = 0, 52  # extended projection
time_points = np.arange(t_start, t_end + dt, dt)
weekly_time = np.arange(0, 52)
steps_by_weeks = int(1 / dt)


def load_best_fit(csv_path, param_order):
    df = pd.read_csv(csv_path).set_index("Parameter")
    return np.array([df.loc[name, "Best_Fit"] for name in param_order])


def expand_single_to_full(single_params):
    single_dict = dict(zip(PARAM_NAMES_SINGLE_SIGMOID_FIT, single_params))
    single_dict["T_perceive"] = T_PERCEIVE_FIXED_SIGMOID
    return np.array([single_dict.get(name, X1_C1_PLACEHOLDER) for name in PARAM_NAMES])


double_best = load_best_fit(
    os.path.join(BASE_DIR, "data", "fits", "double", "vetlyanka_polished_fit.csv"),
    PARAM_NAMES,
)
single_best = expand_single_to_full(
    load_best_fit(
        os.path.join(BASE_DIR, "data", "fits", "single", "vetlyanka_polished_fit_single.csv"),
        PARAM_NAMES_SINGLE_SIGMOID_FIT,
    )
)

sol_double = solve_ivp(plague_model_double, [t_start, t_end], initial_conditions,
                        args=(double_best,), t_eval=time_points, method="RK45")
sol_single = solve_ivp(plague_model_single, [t_start, t_end], initial_conditions,
                        args=(single_best,), t_eval=time_points, method="RK45")

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

ax.plot(time_points[50:], double_rate_smooth[50:], color=COLOR_DOUBLE, linestyle="-", label="Double-sigmoid")
ax.plot(time_points[50:], single_rate_smooth[50:], color=COLOR_SINGLE, linestyle="-", label="Single-sigmoid")
ax.scatter(np.arange(n_weeks_observed), newinfectedweekly, color=COLOR_OBS, marker="o",
           s=14, zorder=5, linewidths=0, label="Observed")
ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=1)
ax.axvline(21, color="gray", linewidth=0.6, linestyle=":", zorder=1)
ax.text(21.3, ax.get_ylim()[1] * 0.92, "end of\nobservation", fontsize=7, color="gray")

ax.set_xlim(15, 52)
ax.set_title("Tail and extended projection, sigmoidal γ,μ pair", fontsize=9, fontweight="bold")
ax.set_xlabel("Time (weeks)")
ax.set_ylabel("Weekly deaths")
ax.legend(frameon=False, loc="upper right", handlelength=1.6, fontsize=7)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, linewidth=0.4, alpha=0.35)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Fig_sigmoidal_tail_zoom.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "Fig_sigmoidal_tail_zoom.png"), dpi=600, bbox_inches="tight")
print(f"Saved {FIG_DIR}/Fig_sigmoidal_tail_zoom.pdf and .png")
plt.show()
plt.close(fig)