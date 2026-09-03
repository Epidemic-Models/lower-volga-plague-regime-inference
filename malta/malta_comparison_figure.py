"""
malta_comparison_figure.py

Main comparison figure for Malta: double-sigmoid vs single-sigmoid fit
against the real, observed cumulative and weekly death counts. Mirrors
plot_sigmoidal_pair_split.py's structure and styling exactly, adapted
for Malta's 27-week series.

DATA LAYOUT: single reads malta/data/fits/single/malta_single_polished_fit.csv.
Double reads malta/data/fits/double/malta_sigmoidal_dr3_polished_fit.csv (the
escape-route-corrected, dispose_rate=3.0-fixed final result -- NOT
malta_double_polished_fit.csv, which is the superseded raw fit).

Requires:
    plague_single_sigmoid_model.py and plague_double_sigmoid_model.py in
    the vetlyanka/ folder (imported via an explicit path below).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

VETLYANKA_DIR = "/Users/bornalid/PycharmProjects/lower-volga-plague-regime-inference/vetlyanka "
sys.path.insert(0, VETLYANKA_DIR)

from plague_single_sigmoid_model import plague_model as plague_model_single
from plague_double_sigmoid_model import plague_model as plague_model_double

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FULL_NAMES_SINGLE = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2",
                     "mu1", "mu2", "T_perceive", "sensitivity", "dispose_rate"]

# Double's final-confirm result is saved WITHOUT dispose_rate (it's fixed,
# not searched) -- this is the reduced 12-name order used in that CSV.
DOUBLE_FINAL_PARAM_NAMES = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2",
                            "mu1", "mu2", "T_perceive", "sensitivity"]
DOUBLE_DISPOSE_RATE_FIXED = 3.0  # matches malta_sigmoidal_dispose_rate_final_confirm.py

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
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "fits", "single",
                                    "malta_single_polished_fit.csv")).set_index("Parameter")
    d = dict(zip(df.index, df["Best_Fit"]))
    return np.array([d.get(n, 1.0) for n in FULL_NAMES_SINGLE])


def load_double_full():
    # NOTE: reads from data/diagnostics/, since that's where
    # malta_sigmoidal_dispose_rate_final_confirm.py currently saves,
    # pre-reorg. Update this path once that script's OUTPUT_DIR is
    # redirected to data/fits/double/.
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "diagnostics",
                                    "malta_sigmoidal_dr3_polished_fit.csv")).set_index("Parameter")
    d = dict(zip(df.index, df["Best_Fit"]))
    d["dispose_rate"] = DOUBLE_DISPOSE_RATE_FIXED
    return np.array([d[n] for n in FULL_NAMES_SINGLE])


single_best = load_single_full()
double_best = load_double_full()

sol_single = solve_ivp(plague_model_single, [t_start, t_end], initial_conditions,
                        args=(single_best,), t_eval=t_points, method="RK45")
sol_double = solve_ivp(plague_model_double, [t_start, t_end], initial_conditions,
                        args=(double_best,), t_eval=t_points, method="RK45")

# DDI (index 4), plague-only -- matches what both models were actually fit against
cumulative_single = sol_single.y[4][::steps_per_week][:n_weeks]
cumulative_double = sol_double.y[4][::steps_per_week][:n_weeks]

weekly_single_smooth = np.gradient(sol_single.y[4], t_points)
weekly_double_smooth = np.gradient(sol_double.y[4], t_points)
weekly_observed = np.diff(np.concatenate(([0.0], observed)))

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