"""
plot_frozen_tail_zoom.py   (figures step -- run after 02_fits)

Zoom on the mortality TAIL and an extended projection to week 52 for the
constant-gamma/mu pair: double-frozen reproduces the real halt (observed 0, 0, 0
in weeks 20-22); single-frozen predicts continuing deaths and a resurgence.
The projection is a diagnostic of each fitted mechanism's long-run behaviour,
not a forecast; observed data exist only through week 22.

PLOTTING POSITIONS (corrected alignment): observed weekly count for week k is
drawn at the week's midpoint t = k - 0.5; the end of observation is t = 22.
Model curves are the smooth death rate dDR/dt (deaths per week).

Both T_perceive values are READ from the fit CSVs.

Writes: <this folder>/results/figures/Fig_frozen_tail_zoom.pdf / .png
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
DOUBLE_CSV = os.path.join(FITS, "double_frozen", "vetlyanka_constant_mu_gamma_polished_fit.csv")
SINGLE_CSV = os.path.join(FITS, "single_frozen", "vetlyanka_constant_mu_gamma_single_FIXED_fit.csv")
LABEL_DOUBLE, LABEL_SINGLE = "Double, constant γ,μ", "Single, constant γ,μ"
TITLE = "Tail and extended projection, constant γ,μ pair"
OUT_NAME = "Fig_frozen_tail_zoom"

initial_conditions = [1699, 1, 0, 0, 0, 0, 0]
weekly_obs = np.array([3, 0, 2, 0, 1, 0, 2, 3, 0, 1, 7, 8, 7, 56, 169, 54, 32, 19, 12, 0, 0, 0])
weeks = np.arange(1, 23)
t_weekly_obs = weeks - 1 if LEGACY_ALIGNMENT else weeks - 0.5
END_OF_OBS = 21 if LEGACY_ALIGNMENT else 22
dt, t_start, t_end = 0.01, 0.0, 52.0
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
    rate_d = np.gradient(sol_d.y[5], time_points)
    rate_s = np.gradient(sol_s.y[5], time_points)

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.labelsize": 10,
                         "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "lines.linewidth": 1.8, "axes.linewidth": 0.9,
                         "xtick.major.width": 0.9, "ytick.major.width": 0.9})
    C_D, C_S, C_O = "#B3222B", "#E08214", "black"
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=300)
    skip = 50
    ax.plot(time_points[skip:], rate_d[skip:], color=C_D, label=LABEL_DOUBLE)
    ax.plot(time_points[skip:], rate_s[skip:], color=C_S, label=LABEL_SINGLE)
    ax.scatter(t_weekly_obs, weekly_obs, color=C_O, marker="o", s=14, zorder=5, linewidths=0, label="Observed")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=1)
    ax.axvline(END_OF_OBS, color="gray", linewidth=0.6, linestyle=":", zorder=1)
    ax.set_xlim(15, 52)
    visible = time_points >= 15
    ymax = max(rate_d[visible].max(), rate_s[visible].max(), weekly_obs[t_weekly_obs >= 15].max()) * 1.1
    ax.set_ylim(-0.03 * ymax, ymax)
    ax.text(END_OF_OBS + 0.3, ymax * 0.92, "end of\nobservation", fontsize=6.5, color="gray")
    ax.set_title(TITLE, fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (weeks)")
    ax.set_ylabel("Weekly deaths")
    ax.legend(frameon=False, loc="upper right", handlelength=1.6, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linewidth=0.4, alpha=0.35)

    plt.tight_layout()
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        plt.savefig(os.path.join(FIG_DIR, f"{OUT_NAME}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"Saved {os.path.join(FIG_DIR, OUT_NAME)}.pdf and .png")


if __name__ == "__main__":
    main()