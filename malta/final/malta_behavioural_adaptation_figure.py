"""
malta_behavioural_adaptation_figure.py

Behavioral-adaptation figure for Malta's double-sigmoid model (the
winning model), mirroring Vetlyanka's behavioural_adaptation_figure.py
exactly: baseline vs. effective transmission over time, plus percent
reduction, using the confirmed, final dispose_rate=3.0 parameters.

Path pattern matches the executed final/diagnostics/ reorganization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VETLYANKA_ROOT = "/Users/bornalid/PycharmProjects/lower-volga-plague-regime-inference/vetlyanka "
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

import sys
sys.path.insert(0, VETLYANKA_ROOT)
from plague_double_sigmoid_model import plague_model, double_sigmoid

t_start, t_end, dt = 0, 26, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
initial_conditions = [11000, 1, 0, 0, 0, 0, 0]

# Confirmed, final Malta double-sigmoid parameters (dispose_rate=3.0)
params = np.array([2.417914, 8.441342, 9.100074, 21.691271, 1.011883, 0.452404,
                    0.700000, 1.000000, 1.400000, 1.000000, 2.433220, 0.004729, 3.0])
b1, b2, x0, x1, c, c1, gamma1, gamma2, mu1, mu2, T_perceive, sensitivity, dispose_rate = params

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 9, "axes.labelsize": 10,
    "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "lines.linewidth": 1.8, "axes.linewidth": 0.9,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
})

sol = solve_ivp(plague_model, [t_start, t_end], initial_conditions,
                 args=(params,), t_eval=t_points, method="RK45")
I = sol.y[1]
P = sol.y[6]

beta_base = double_sigmoid(t_points, b1, b2, x0, x1, c, c1)
suppression = np.exp(-sensitivity * P)
beta_eff = beta_base * suppression
reduction_pct = (beta_base - beta_eff) / beta_base * 100

peak_idx = np.argmax(reduction_pct)
peak_reduction = reduction_pct[peak_idx]
peak_week = t_points[peak_idx]
IWM = np.sum(reduction_pct * I) / np.sum(I)
print(f"Malta double-sigmoid: peak reduction = {peak_reduction:.1f}% at week {peak_week:.1f}, IWM = {IWM:.1f}%")

fig, ax1 = plt.subplots(figsize=(3.42, 2.6), dpi=300)
ax1.set_title("Malta, Double-Sigmoid Model", fontsize=10, fontweight="bold")
ax1.plot(t_points, beta_base, color="#B3222B", linestyle="--", label=r"Baseline $\beta_{\mathrm{base}}(t)$")
ax1.plot(t_points, beta_eff, color="#B3222B", linestyle="-", label=r"Effective $\beta_{\mathrm{eff}}(t)$")
ax1.set_xlabel("Time (weeks)")
ax1.set_ylabel(r"$\beta(t)$ (week$^{-1}$)")
ax1.spines[["top"]].set_visible(False)

ax2 = ax1.twinx()
ax2.plot(t_points, reduction_pct, color="gray", linestyle="-.", linewidth=1.2, label="Reduction (%)")
ax2.scatter([peak_week], [peak_reduction], color="gray", marker="o", s=18, zorder=5)
ax2.annotate(f"{peak_reduction:.1f}%", (peak_week, peak_reduction),
             textcoords="offset points", xytext=(6, 4), fontsize=7, color="gray")
ax2.set_ylabel("Reduction in transmission (%)", color="gray")
ax2.tick_params(axis="y", labelcolor="gray")
ax2.set_ylim(0, max(reduction_pct.max() * 1.25, 10))
ax2.spines[["top"]].set_visible(False)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left", fontsize=7, handlelength=1.6)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Fig_malta_behavioural_adaptation_double.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "Fig_malta_behavioural_adaptation_double.png"), dpi=600, bbox_inches="tight")
print(f"Saved {FIG_DIR}/Fig_malta_behavioural_adaptation_double.pdf and .png")
plt.show()
plt.close(fig)