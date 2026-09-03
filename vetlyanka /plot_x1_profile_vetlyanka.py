"""
plot_x1_profile_sse.py

Publication-quality plot of the profile-SSE analysis for the
decline-onset parameter x1 in the standalone Vetlyanka double-regime model.

This script does NOT perform any optimization.
It reads the already-computed profile results from:

    x1_profile_sse_refined.csv

and saves the figure in:

    figures/Fig_x1_profile_sse.pdf
    figures/Fig_x1_profile_sse.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FIG_DIR = os.path.join(
    BASE_DIR,
    "figures"
)

os.makedirs(
    FIG_DIR,
    exist_ok=True
)

PROFILE_FILE = os.path.join(
    BASE_DIR,
    "x1_profile_sse_refined.csv"
)


# ============================================================
# UNRESTRICTED FIT
# ============================================================

# Best-fit value from the unrestricted polished double-regime fit.
UNRESTRICTED_X1 = 17.49


# ============================================================
# PLOT STYLE
# ============================================================

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


# ============================================================
# LOAD PROFILE RESULTS
# ============================================================

if not os.path.exists(PROFILE_FILE):
    raise FileNotFoundError(
        f"Could not find profile results:\n{PROFILE_FILE}"
    )

profile_df = pd.read_csv(PROFILE_FILE)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "x1",
    "profile_sse",
]

missing_columns = [
    column
    for column in required_columns
    if column not in profile_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in profile file: "
        f"{missing_columns}"
    )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

profile_df = profile_df[
    np.isfinite(profile_df["x1"])
    &
    np.isfinite(profile_df["profile_sse"])
].copy()

profile_df = profile_df.sort_values(
    "x1"
).reset_index(drop=True)


# ============================================================
# CALCULATE DELTA SSE
# ============================================================

profile_min_sse = profile_df[
    "profile_sse"
].min()

profile_df["delta_sse_plot"] = (
    profile_df["profile_sse"]
    - profile_min_sse
)


# ============================================================
# FIND PROFILE MINIMUM
# ============================================================

best_idx = profile_df[
    "profile_sse"
].idxmin()

best_x1 = profile_df.loc[
    best_idx,
    "x1"
]

best_sse = profile_df.loc[
    best_idx,
    "profile_sse"
]


# ============================================================
# PRINT SUMMARY
# ============================================================

print("=" * 65)
print("VETLYANKA x1 PROFILE-SSE")
print("=" * 65)

print(
    f"\nProfile minimum x1 = "
    f"{best_x1:.2f} weeks"
)

print(
    f"Profile minimum SSE = "
    f"{best_sse:.6f}"
)

print(
    f"Unrestricted fit x1 = "
    f"{UNRESTRICTED_X1:.2f} weeks"
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.0, 3.4),
    dpi=300
)


# ============================================================
# PROFILE CURVE
# ============================================================

ax.plot(
    profile_df["x1"],
    profile_df["delta_sse_plot"],
    linestyle="-",
    marker="o",
    markersize=3.5,
    markeredgewidth=0.6,
    label="Profile SSE"
)


# ============================================================
# PROFILE MINIMUM
# ============================================================

ax.axvline(
    best_x1,
    linestyle="--",
    linewidth=1.2,
    label=(
        rf"Profile minimum "
        rf"$x_1={best_x1:.2f}$"
    )
)


# ============================================================
# UNRESTRICTED BEST FIT
# ============================================================

ax.axvline(
    UNRESTRICTED_X1,
    linestyle=":",
    linewidth=1.2,
    label=(
        rf"Unrestricted fit "
        rf"$x_1={UNRESTRICTED_X1:.2f}$"
    )
)


# ============================================================
# AXIS LABELS
# ============================================================

ax.set_xlabel(
    r"Fixed decline-onset parameter $x_1$ (weeks)"
)

ax.set_ylabel(
    r"$\Delta$SSE relative to profile minimum"
)


# ============================================================
# AXIS LIMITS
# ============================================================

ax.set_xlim(
    profile_df["x1"].min(),
    profile_df["x1"].max()
)

ax.set_ylim(
    bottom=0
)


# ============================================================
# TICKS
# ============================================================

ax.set_xticks(
    np.arange(
        np.ceil(profile_df["x1"].min()),
        np.floor(profile_df["x1"].max()) + 1,
        1
    )
)

ax.tick_params(
    direction="out",
    length=4
)


# ============================================================
# LEGEND
# ============================================================

ax.legend(
    frameon=False,
    loc="upper right",
    handlelength=2.0
)


# ============================================================
# FIGURE STYLE
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    True,
    linewidth=0.4,
    alpha=0.30
)


# ============================================================
# SAVE FIGURE
# ============================================================

plt.tight_layout()

PDF_FILE = os.path.join(
    FIG_DIR,
    "Fig_x1_profile_sse.pdf"
)

PNG_FILE = os.path.join(
    FIG_DIR,
    "Fig_x1_profile_sse.png"
)

plt.savefig(
    PDF_FILE,
    bbox_inches="tight"
)

plt.savefig(
    PNG_FILE,
    dpi=600,
    bbox_inches="tight"
)

print(
    f"\nSaved "
    f"{FIG_DIR}/Fig_x1_profile_sse.pdf and .png"
)

plt.show()

plt.close(fig)