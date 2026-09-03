"""
malta_plot_x1_profile_sse.py

Publication-quality plot of the profile-SSE analysis for the
decline-onset parameter x1 in Malta's double-regime model.

Mirrors plot_x1_profile_sse.py (Vetlyanka) exactly, adapted for Malta's
saved profile results and confirmed unrestricted fit. Does NOT perform
any optimization -- reads the already-computed profile from
malta_x1_profile_sse.csv (saved by malta_x1_profile_sse.py).

Unlike Vetlyanka's profile (single, clean minimum), Malta's profile
shows a real secondary local minimum/plateau around x1=16.6-17.8 --
this figure shows that honestly, not smoothed over.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

PROFILE_FILE = os.path.join(PROJECT_ROOT, "data", "diagnostics", "malta_x1_profile_sse.csv")

# Confirmed, final unrestricted double-sigmoid fit (dispose_rate=3.0)
UNRESTRICTED_X1 = 21.692

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 9, "axes.labelsize": 10,
    "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "lines.linewidth": 1.8, "axes.linewidth": 0.9,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
})

if not os.path.exists(PROFILE_FILE):
    raise FileNotFoundError(f"Could not find profile results:\n{PROFILE_FILE}")

profile_df = pd.read_csv(PROFILE_FILE)

required_columns = ["x1", "SSE"]
missing_columns = [c for c in required_columns if c not in profile_df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in profile file: {missing_columns}")

profile_df = profile_df[np.isfinite(profile_df["x1"]) & np.isfinite(profile_df["SSE"])].copy()
profile_df = profile_df.sort_values("x1").reset_index(drop=True)

profile_min_sse = profile_df["SSE"].min()
profile_df["delta_sse_plot"] = profile_df["SSE"] - profile_min_sse

best_idx = profile_df["SSE"].idxmin()
best_x1 = profile_df.loc[best_idx, "x1"]
best_sse = profile_df.loc[best_idx, "SSE"]

print("=" * 65)
print("MALTA x1 PROFILE-SSE")
print("=" * 65)
print(f"\nProfile minimum x1 = {best_x1:.2f} weeks")
print(f"Profile minimum SSE = {best_sse:.6f}")
print(f"Unrestricted fit x1 = {UNRESTRICTED_X1:.2f} weeks")

fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=300)

ax.plot(profile_df["x1"], profile_df["delta_sse_plot"], linestyle="-", marker="o",
        markersize=3.5, markeredgewidth=0.6, color="#B3222B", label="Profile SSE")

ax.axvline(best_x1, linestyle="--", linewidth=1.2, color="black",
           label=rf"Profile minimum $x_1={best_x1:.2f}$")
ax.axvline(UNRESTRICTED_X1, linestyle=":", linewidth=1.2, color="gray",
           label=rf"Unrestricted fit $x_1={UNRESTRICTED_X1:.2f}$")

ax.set_xlabel(r"Fixed decline-onset parameter $x_1$ (weeks)")
ax.set_ylabel(r"$\Delta$SSE relative to profile minimum")
ax.set_xlim(profile_df["x1"].min(), profile_df["x1"].max())
ax.set_ylim(bottom=0)
ax.set_xticks(np.arange(np.ceil(profile_df["x1"].min()), np.floor(profile_df["x1"].max()) + 1, 2))
ax.tick_params(direction="out", length=4)
ax.legend(frameon=False, loc="upper center", handlelength=2.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linewidth=0.4, alpha=0.30)

plt.tight_layout()
PDF_FILE = os.path.join(FIG_DIR, "Fig_malta_x1_profile_sse.pdf")
PNG_FILE = os.path.join(FIG_DIR, "Fig_malta_x1_profile_sse.png")
plt.savefig(PDF_FILE, bbox_inches="tight")
plt.savefig(PNG_FILE, dpi=600, bbox_inches="tight")
print(f"\nSaved {FIG_DIR}/Fig_malta_x1_profile_sse.pdf and .png")
plt.show()
plt.close(fig)