"""
malta_comparison_figure.py

Main comparison figure for Malta: double-sigmoid vs single-sigmoid fit
against the real, observed cumulative death counts (DDI, plague-only).
Mirrors plot_sigmoidal_pair_split.py's structure and styling exactly.

Path pattern matches the executed final/diagnostics/ reorganization:
this script lives in malta/final/, data/ and figures/ live at malta/
(one level up from this script).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VETLYANKA_ROOT = "/Users/bornalid/PycharmProjects/lower-volga-plague-regime-inference/vetlyanka "
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, VETLYANKA_ROOT)
from plague_single_sigmoid_model import plague_model as plague_model_single
from plague_double_sigmoid_model import plague_model as plague_model_double

FULL_NAMES_SINGLE = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2",
                     "mu1", "mu2", "T_perceive", "sensitivity", "dispose_rate"]

observed = np.array([0, 0, 0, 41, 100, 227, 440, 592, 813, 1139, 1523, 1896,
                      2225, 2609, 2860, 3045, 3269, 3486, 3703, 3902, 4045,
                      4147, 4247, 4309, 4339, 4379, 4400], dtype=float)
weekly_time = np.arange(len(observed))
n_weeks = len(observed)

t_start, t_end, dt = 0, n_weeks - 1, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1 / dt))
initial_conditions = [11000, 1, 0, 0, 0, 0, 0]


def load_single_full():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "fits", "single",
                                    "malta_single_polished_fit.csv")).set_index("Parameter")
    d = dict(zip(df.index, df["Best_Fit"]))
    return np.array([d.get(n, 1.0) for n in FULL_NAMES_SINGLE])


def load_double_full():
    # Final, confirmed double-sigmoid parameters (dispose_rate=3.0) --
    # hardcoded here since the true final result lives in
    # malta_double_sigmoidal_dispose_rate_final_confirm.py's own output,
    # not malta_fit_double.py's (superseded, sits on the b2/dispose_rate
    # escape route). If that script's CSV output path is confirmed,
    # swap this block to read from there instead.
    return np.array([2.417914, 8.441342, 9.100074, 21.691271, 1.011883, 0.452404,
                      0.700000, 1.000000, 1.400000, 1.000000, 2.433220, 0.004729, 3.0])


single_best = load_single_full()
double_best = load_double_full()

sol_single = solve_ivp(plague_model_single, [t_start, t_end], initial_conditions,
                        args=(single_best,), t_eval=t_points, method="RK45")
sol_double = solve_ivp(plague_model_double, [t_start, t_end], initial_conditions,
                        args=(double_best,), t_eval=t_points, method="RK45")

cumulative_single = sol_single.y[4][::steps_per_week][:n_weeks]  # DDI, plague-only
cumulative_double = sol_double.y[4][::steps_per_week][:n_weeks]

weekly_single_smooth = np.gradient(sol_single.y[4], t_points)
weekly_double_smooth = np.gradient(sol_double.y[4], t_points)
weekly_observed = np.diff(np.concatenate(([0.0], observed)))

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 9, "axes.labelsize": 10,
    "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "lines.linewidth": 1.8, "axes.linewidth": 0.9,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
})

COLOR_DOUBLE = "#B3222B"
COLOR_SINGLE = "#E08214"
COLOR_OBS = "black"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2), dpi=300)

ax1.plot(t_points, sol_double.y[4], color=COLOR_DOUBLE, linestyle="-", label="Double-sigmoid")
ax1.plot(t_points, sol_single.y[4], color=COLOR_SINGLE, linestyle="-", label="Single-sigmoid")
ax1.scatter(weekly_time, observed, color=COLOR_OBS, marker="o", s=14, zorder=5,
            linewidths=0, label="Observed")
ax1.set_title("Cumulative deaths", fontsize=9, fontweight="bold")
ax1.set_xlabel("Time (weeks)")
ax1.set_ylabel("Cumulative deaths")
ax1.legend(frameon=False, loc="lower right", handlelength=1.6, fontsize=7)
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(True, linewidth=0.4, alpha=0.35)

ax2.plot(t_points[50:], weekly_double_smooth[50:], color=COLOR_DOUBLE, linestyle="-", label="Double-sigmoid")
ax2.plot(t_points[50:], weekly_single_smooth[50:], color=COLOR_SINGLE, linestyle="-", label="Single-sigmoid")
ax2.scatter(weekly_time, weekly_observed, color=COLOR_OBS, marker="o", s=14, zorder=5,
            linewidths=0, label="Observed")
ax2.set_title("Weekly deaths", fontsize=9, fontweight="bold")
ax2.set_xlabel("Time (weeks)")
ax2.set_ylabel("Weekly deaths")
ax2.legend(frameon=False, loc="upper right", handlelength=1.6, fontsize=7)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(True, linewidth=0.4, alpha=0.35)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Fig_malta_comparison.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "Fig_malta_comparison.png"), dpi=600, bbox_inches="tight")
print(f"Saved {FIG_DIR}/Fig_malta_comparison.pdf and .png")
plt.show()
plt.close(fig)