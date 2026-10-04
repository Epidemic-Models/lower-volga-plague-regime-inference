"""
plot_x1_profile_vetlyanka.py   (figures step -- run after 06_robustness/profile_x1_vetlyanka_refined.py)

Plot of the profile SSE for the decline-onset parameter x1 in the double-sigmoid
model. No optimization here -- it only reads results that already exist:

    06_robustness/results/x1_profile/x1_profile_sse_refined.csv   (the profile)
    02_fits/results/double/vetlyanka_polished_fit.csv            (unrestricted x1)

The unrestricted x1 is READ from the double fit CSV, not typed in.

Writes: <this folder>/results/figures/Fig_x1_profile_sse.pdf / .png
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):  # walk up to vetlyanka/, which contains src/
    if os.path.dirname(ROOT) == ROOT:
        raise FileNotFoundError("Could not find the src/ folder above this script.")
    ROOT = os.path.dirname(ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROFILE_FILE = os.path.join(ROOT, "06_robustness", "results", "x1_profile", "x1_profile_sse_refined.csv")
DOUBLE_CSV = os.path.join(ROOT, "02_fits", "results", "double", "vetlyanka_polished_fit.csv")
FIG_DIR = os.path.join(HERE, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
OUT_NAME = "Fig_x1_profile_sse"


def main():
    for path, step in ((PROFILE_FILE, "06_robustness/profile_x1_vetlyanka_refined.py"),
                       (DOUBLE_CSV, "02_fits double fit")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found -- run {step} first.")

    fit = pd.read_csv(DOUBLE_CSV)
    unrestricted_x1 = float(fit.loc[fit["Parameter"] == "x1", "Best_Fit"].iloc[0])

    df = pd.read_csv(PROFILE_FILE)
    missing = [c for c in ("x1", "profile_sse") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in profile file: {missing}")
    df = df[np.isfinite(df["x1"]) & np.isfinite(df["profile_sse"])].sort_values("x1").reset_index(drop=True)

    best = df["profile_sse"].idxmin()
    best_x1, best_sse = df.loc[best, "x1"], df.loc[best, "profile_sse"]
    df["delta_sse_plot"] = df["profile_sse"] - best_sse

    print("=" * 65)
    print("VETLYANKA x1 PROFILE-SSE")
    print("=" * 65)
    print(f"\nProfile minimum x1  = {best_x1:.2f} weeks")
    print(f"Profile minimum SSE = {best_sse:.6f}")
    print(f"Unrestricted fit x1 = {unrestricted_x1:.2f} weeks (from {os.path.basename(DOUBLE_CSV)})")

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.labelsize": 10,
                         "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "lines.linewidth": 1.8, "axes.linewidth": 0.9,
                         "xtick.major.width": 0.9, "ytick.major.width": 0.9})
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=300)
    ax.plot(df["x1"], df["delta_sse_plot"], linestyle="-", marker="o", markersize=3.5,
            markeredgewidth=0.6, label="Profile SSE")
    ax.axvline(best_x1, linestyle="--", linewidth=1.2, color="C1",
               label=rf"Profile minimum $x_1={best_x1:.2f}$")
    ax.axvline(unrestricted_x1, linestyle=":", linewidth=1.2, color="C2",
               label=rf"Unrestricted fit $x_1={unrestricted_x1:.2f}$")
    ax.set_xlabel(r"Fixed decline-onset parameter $x_1$ (weeks)")
    ax.set_ylabel(r"$\Delta$SSE relative to profile minimum")
    ax.set_xlim(df["x1"].min(), df["x1"].max())
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(np.ceil(df["x1"].min()), np.floor(df["x1"].max()) + 1, 1))
    ax.tick_params(direction="out", length=4)
    leg = ax.legend(frameon=True, framealpha=1.0, loc="upper right", handlelength=2.0)
    leg.get_frame().set_edgecolor("none")   # white box so the vertical lines don't run through the text
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linewidth=0.4, alpha=0.30)

    plt.tight_layout()
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        plt.savefig(os.path.join(FIG_DIR, f"{OUT_NAME}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"\nSaved {os.path.join(FIG_DIR, OUT_NAME)}.pdf and .png")


if __name__ == "__main__":
    main()