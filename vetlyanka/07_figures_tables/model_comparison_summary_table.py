"""
model_comparison_summary_table.py   (step 08 -- run after 03_model_comparison)

Builds LaTeX tables of SSE, k, AIC, AICc and BIC for all five models
(cumulative basis, plus the incremental/weekly basis as a second table).
Reads the CSVs saved by 03_model_comparison/AIC_AICc_BIC_gamma_mu_freeze.py
rather than recomputing, so the table always matches that script's latest run.

Reads:  03_model_comparison/results/
Writes: 07_figures_tables/results/model_comparison_summary_table.tex
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # .../vetlyanka/07_figures_tables
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):  # walk up to vetlyanka/, which contains src/
    if os.path.dirname(ROOT) == ROOT:
        raise FileNotFoundError("Could not find the src/ folder above this script.")
    ROOT = os.path.dirname(ROOT)

import pandas as pd

COMPARISONS_DIR = os.path.join(ROOT, "03_model_comparison", "results")
OUTPUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CUM_CSV = os.path.join(COMPARISONS_DIR, "vetlyanka_model_comparison_with_frozen_single.csv")
INCR_CSV = os.path.join(COMPARISONS_DIR, "vetlyanka_model_comparison_incremental_with_frozen_single.csv")

# Map the CSV row labels to the paper's model names
MODEL_LABELS = {
    "Null (no feedback)": "Null",
    "Single-sigmoid (+feedback)": "Single",
    "Double-sigmoid": "Double",
    "Double, frozen \u03b3, \u03bc": "Double, constant $\\gamma,\\mu$",
    "Single, frozen \u03b3, \u03bc": "Single, constant $\\gamma,\\mu$",
}

MODEL_ORDER = ["Null", "Single", "Double", "Double, constant $\\gamma,\\mu$", "Single, constant $\\gamma,\\mu$"]


def load_and_relabel(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found -- run 03_model_comparison/AIC_AICc_BIC_gamma_mu_freeze.py first.")
    df = pd.read_csv(csv_path, index_col=0)
    df.index = [MODEL_LABELS.get(name.replace(" (incremental)", "").replace(", incremental", ""), name)
                for name in df.index]
    return df.loc[MODEL_ORDER]


def build_table(basis_label, df, label_suffix):
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \small",
        rf"    \caption{{Model comparison summary, {basis_label} basis. "
        r"SSE: sum of squared errors. $k$: number of free parameters. "
        r"Lower AIC/AICc/BIC indicates a better balance of fit against complexity.}",
        r"    \begin{tabular}{lccccc}",
        r"        \toprule",
        r"        \textbf{Model} & \textbf{$k$} & \textbf{SSE} & \textbf{AIC} & \textbf{AICc} & \textbf{BIC} \\",
        r"        \midrule",
    ]
    for name in MODEL_ORDER:
        row = df.loc[name]
        lines.append(f"        {name} & {int(row['k'])} & {row['SSE']:.1f} & "
                     f"{row['AIC']:.2f} & {row['AICc']:.2f} & {row['BIC']:.2f} \\\\")
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        rf"    \label{{tab:model_comparison_summary_{label_suffix}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    cum_table = build_table("cumulative", load_and_relabel(CUM_CSV), "cum")
    incr_table = build_table("incremental (weekly)", load_and_relabel(INCR_CSV), "incr")
    full_tex = cum_table + "\n\n" + incr_table
    print(full_tex)

    out_path = os.path.join(OUTPUT_DIR, "model_comparison_summary_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_tex)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()