"""
plot_sigmoidal_pair_split.py   (figures step -- run after 02_fits)

One figure, two panels: cumulative deaths (left) and weekly deaths (right),
double-sigmoid and single-sigmoid (time-varying gamma, mu) overlaid, against the
observed data.

PLOTTING POSITIONS (corrected alignment): the observed cumulative count for
week k is the value at the END of week k, so it is drawn at t = k (k = 1..22).
The observed weekly count for week k covers t in [k-1, k], so it is drawn at the
week's midpoint t = k - 0.5, where the smooth model death-rate curve
(dDR/dt, deaths per week) should match it.

Both fixed/fitted T_perceive values are READ from the fit CSVs.

Writes: <this folder>/results/figures/Fig_sigmoidal_pair_grid.pdf / .png
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):  # walk up to vetlyanka/, which contains src/
    if os.path.dirname(ROOT) == ROOT:
        raise FileNotFoundError("Could not find the src/ folder above this script.")
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from plague_double_sigmoid_model import plague_model as plague_model_double
from plague_single_sigmoid_model import plague_model as plague_model_single
from vetlyanka_bounds import PARAM_NAMES

LEGACY_ALIGNMENT = False
FITS = os.path.join(ROOT, "02_fits", "results")
FIG_DIR = os.path.join(HERE, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
DOUBLE_CSV = os.path.join(FITS, "double", "vetlyanka_polished_fit.csv")
SINGLE_CSV = os.path.join(FITS, "single", "vetlyanka_polished_fit_single.csv")
LABEL_DOUBLE, LABEL_SINGLE = "Double-sigmoid", "Single-sigmoid"
OUT_NAME = "Fig_sigmoidal_pair_grid"

initial_conditions = [1699, 1, 0, 0, 0, 0, 0]
weekly_obs = np.array([3, 0, 2, 0, 1, 0, 2, 3, 0, 1, 7, 8, 7, 56, 169, 54, 32, 19, 12, 0, 0, 0])
cumulative_obs = np.cumsum(weekly_obs)
weeks = np.arange(1, 23)
t_cum_obs = weeks - 1 if LEGACY_ALIGNMENT else weeks          # end of week k
t_weekly_obs = weeks - 1 if LEGACY_ALIGNMENT else weeks - 0.5  # middle of week k
dt, t_start, t_end = 0.01, 0.0, 22.0
time_points = np.arange(t_start, t_end + dt / 2, dt)


def load_full(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run the matching 02_fits script first.")
    df = pd.read_csv(path)
    d = dict(zip(df["Parameter"], df["Best_Fit"].astype(float)))
    if "gamma_const" in d:
        d["gamma1"] = d["gamma2"] = d["gamma_const"]
        d["mu1"] = d["mu2"] = d["mu_const"]
    return np.array([d.get(n, 1.0) for n in PARAM_NAMES], dtype=float)


def main():
    sol_d = solve_ivp(plague_model_double, [t_start, t_end], initial_conditions,
                      args=(load_full(DOUBLE_CSV),), t_eval=time_points, method="RK45")
    sol_s = solve_ivp(plague_model_single, [t_start, t_end], initial_conditions,
                      args=(load_full(SINGLE_CSV),), t_eval=time_points, method="RK45")
    DR_d, DR_s = sol_d.y[5], sol_s.y[5]
    rate_d, rate_s = np.gradient(DR_d, time_points), np.gradient(DR_s, time_points)

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.labelsize": 10,
                         "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "lines.linewidth": 1.8, "lines.markersize": 4,
                         "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9})
    C_D, C_S, C_O = "#B3222B", "#E08214", "black"
    fig, (ax_cum, ax_wk) = plt.subplots(1, 2, figsize=(7.0, 3.2), dpi=300)

    ax_cum.plot(time_points, DR_d, color=C_D, label=LABEL_DOUBLE)
    ax_cum.plot(time_points, DR_s, color=C_S, label=LABEL_SINGLE)
    ax_cum.scatter(t_cum_obs, cumulative_obs, color=C_O, marker="o", s=14, zorder=5, linewidths=0, label="Observed")
    ax_cum.set_title("Cumulative deaths", fontsize=9, fontweight="bold")
    ax_cum.set_ylabel("Cumulative deaths")

    skip = 50   # first 0.5 week: solver start-up transient in the derivative, not shown
    ax_wk.plot(time_points[skip:], rate_d[skip:], color=C_D, label=LABEL_DOUBLE)
    ax_wk.plot(time_points[skip:], rate_s[skip:], color=C_S, label=LABEL_SINGLE)
    ax_wk.scatter(t_weekly_obs, weekly_obs, color=C_O, marker="o", s=14, zorder=5, linewidths=0, label="Observed")
    ax_wk.set_title("Weekly deaths", fontsize=9, fontweight="bold")
    ax_wk.set_ylabel("Weekly deaths")

    for ax in (ax_cum, ax_wk):
        ax.set_xlabel("Time (weeks)")
        ax.set_xlim(0, 22.5)
        ax.legend(frameon=False, loc="upper left", handlelength=1.6, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, linewidth=0.4, alpha=0.35)

    plt.tight_layout()
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        plt.savefig(os.path.join(FIG_DIR, f"{OUT_NAME}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"Saved {os.path.join(FIG_DIR, OUT_NAME)}.pdf and .png")


if __name__ == "__main__":
    main()