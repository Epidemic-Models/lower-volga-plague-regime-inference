"""
behavioural_adaptation_figure.py

Behavioral-feedback figures for all four gamma/mu-varying and constant-
gamma/mu models: baseline vs. effective transmission rate over time,
plus the percent reduction in transmission driven by behavioral
feedback, with the peak reduction and infection-weighted mean (IWM)
printed for each.

TERMINOLOGY: "constant gamma,mu" here (matching Reviewer #1's own
phrasing: "how does the model perform with constant rates") is the
same model previously called "frozen" internally -- data/fits/
folder names keep the old "frozen" naming (renaming now would break
every other script), but all figure titles/labels/output filenames
use the reviewer-aligned "constant gamma,mu" phrasing.

T_perceive: both single-sigmoid variants FIX T_perceive (borrowed from the
double-regime model of the same gamma/mu family). The fixed values are READ
from their fit CSVs ("fixed" rows), not typed in.

Reads the four fits from 02_fits/results/.
Writes (to <this folder>/results/figures/):
    Fig_behavioural_adaptation_double.pdf / .png
    Fig_behavioural_adaptation_single.pdf / .png
    Fig_behavioural_adaptation_double_constant.pdf / .png
    Fig_behavioural_adaptation_single_constant.pdf / .png
and behavioural_adaptation_summary.csv (peak reduction, its week, IWM per model).
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

from plague_double_sigmoid_model import plague_model as plague_model_double, double_sigmoid as double_sigmoid_fn
from plague_single_sigmoid_model import plague_model as plague_model_single, double_sigmoid as single_sigmoid_fn
from vetlyanka_bounds import PARAM_NAMES

X1_C1_PLACEHOLDER = 1.0
FITS = os.path.join(ROOT, "02_fits", "results")
FIG_DIR = os.path.join(HERE, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FIT_CSVS = {
    "double": os.path.join(FITS, "double", "vetlyanka_polished_fit.csv"),
    "single": os.path.join(FITS, "single", "vetlyanka_polished_fit_single.csv"),
    "double_constant": os.path.join(FITS, "double_frozen", "vetlyanka_constant_mu_gamma_polished_fit.csv"),
    "single_constant": os.path.join(FITS, "single_frozen", "vetlyanka_constant_mu_gamma_single_FIXED_fit.csv"),
}


def load_full(path):
    """Any of the four fit CSVs -> 13-slot parameter vector (fixed T_perceive read from the file)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run the matching 02_fits script first.")
    df = pd.read_csv(path)
    d = dict(zip(df["Parameter"], df["Best_Fit"].astype(float)))
    if "gamma_const" in d:
        d["gamma1"] = d["gamma2"] = d["gamma_const"]
        d["mu1"] = d["mu2"] = d["mu_const"]
    if "T_perceive" not in d:
        raise ValueError(f"No T_perceive row in {path} -- re-run that fit with the new 02_fits script.")
    return np.array([d.get(n, X1_C1_PLACEHOLDER) for n in PARAM_NAMES], dtype=float)


t_start, t_end, dt = 0, 22, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]

double_best = load_full(FIT_CSVS["double"])
single_best_full = load_full(FIT_CSVS["single"])
double_constant_best = load_full(FIT_CSVS["double_constant"])
single_constant_best = load_full(FIT_CSVS["single_constant"])
SUMMARY = []

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


def make_behavioural_adaptation_figure(model_fn, beta_base_fn, params, color, out_name, model_label):
    b1, b2, x0, x1, c, c1, gamma1, gamma2, mu1, mu2, T_perceive, sensitivity, dispose_rate = params
    sol = solve_ivp(model_fn, [t_start, t_end], initial_conditions,
                     args=(params,), t_eval=t_points, method="RK45")
    I = sol.y[1]
    P = sol.y[6]

    beta_base = beta_base_fn(t_points, b1, b2, x0, x1, c, c1)
    suppression = np.exp(-sensitivity * P)
    beta_eff = beta_base * suppression
    reduction_pct = (beta_base - beta_eff) / beta_base * 100

    peak_idx = np.argmax(reduction_pct)
    peak_reduction = reduction_pct[peak_idx]
    peak_week = t_points[peak_idx]
    IWM = np.sum(reduction_pct * I) / np.sum(I)
    print(f"{model_label}: peak reduction = {peak_reduction:.1f}% at week {peak_week:.1f}, IWM = {IWM:.1f}%")
    SUMMARY.append({"model": model_label, "peak_reduction_pct": peak_reduction,
                    "peak_week": peak_week, "IWM_pct": IWM, "T_perceive": T_perceive})

    fig, ax1 = plt.subplots(figsize=(3.42, 2.6), dpi=300)
    ax1.set_title(model_label, fontsize=10, fontweight="bold")
    ax1.plot(t_points, beta_base, color=color, linestyle="--", label=r"Baseline $\beta_{\mathrm{base}}(t)$")
    ax1.plot(t_points, beta_eff, color=color, linestyle="-", label=r"Effective $\beta_{\mathrm{eff}}(t)$")
    ax1.set_xlabel("Time (weeks)")
    ax1.set_ylabel(r"$\beta(t)$ (week$^{-1}$)")
    ax1.spines[["top"]].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(t_points, reduction_pct, color="gray", linestyle="-.", linewidth=1.2,
             label="Reduction (%)")
    ax2.scatter([peak_week], [peak_reduction], color="gray", marker="o", s=18, zorder=5)
    ax2.annotate(f"{peak_reduction:.1f}%", (peak_week, peak_reduction),
                 textcoords="offset points", xytext=(6, 4), fontsize=7, color="gray")
    ax2.set_ylabel("Reduction in transmission (%)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(0, max(reduction_pct.max() * 1.25, 10))
    ax2.spines[["top"]].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left",
               fontsize=7, handlelength=1.6)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{out_name}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, f"{out_name}.png"), dpi=600, bbox_inches="tight")
    print(f"Saved {FIG_DIR}/{out_name}.pdf and .png")
    plt.close(fig)


COLOR_DOUBLE = "#B3222B"
COLOR_SINGLE = "#E08214"

make_behavioural_adaptation_figure(
    plague_model_double, double_sigmoid_fn, double_best, COLOR_DOUBLE,
    "Fig_behavioural_adaptation_double", "Double-Sigmoid Model"
)
make_behavioural_adaptation_figure(
    plague_model_single, single_sigmoid_fn, single_best_full, COLOR_SINGLE,
    "Fig_behavioural_adaptation_single", "Single-Sigmoid Model"
)
make_behavioural_adaptation_figure(
    plague_model_double, double_sigmoid_fn, double_constant_best, COLOR_DOUBLE,
    "Fig_behavioural_adaptation_double_constant", "Double-Sigmoid Model, Constant \u03b3,\u03bc"
)
make_behavioural_adaptation_figure(
    plague_model_single, single_sigmoid_fn, single_constant_best, COLOR_SINGLE,
    "Fig_behavioural_adaptation_single_constant", "Single-Sigmoid Model, Constant \u03b3,\u03bc"
)

pd.DataFrame(SUMMARY).to_csv(os.path.join(FIG_DIR, "behavioural_adaptation_summary.csv"), index=False)
print(f"Saved {os.path.join(FIG_DIR, 'behavioural_adaptation_summary.csv')}")