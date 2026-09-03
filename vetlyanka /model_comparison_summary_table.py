"""
model_comparison_summary_table.py

Generates a single LaTeX table showing SSE, k, AIC, AICc, and BIC for
all five models side by side, cumulative basis, for the SI (or main
text if preferred). Reads directly from AIC_AICc_BIC_gamma_mu_freeze.py's
saved output rather than recomputing, so this table always matches
whatever that script most recently produced.

DATA LAYOUT: reads from data/comparisons/.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPARISONS_DIR = os.path.join(BASE_DIR, "data", "comparisons")

CUM_CSV = os.path.join(COMPARISONS_DIR, "vetlyanka_model_comparison_with_frozen_single.csv")
INCR_CSV = os.path.join(COMPARISONS_DIR, "vetlyanka_model_comparison_incremental_with_frozen_single.csv")

# Map the CSV's row index labels to the paper's official model names
MODEL_LABELS = {
    "Null (no feedback)": "Null",
    "Single-sigmoid (+feedback)": "Single",
    "Double-sigmoid": "Double",
    "Double, frozen \u03b3, \u03bc": "Double, constant $\\gamma,\\mu$",
    "Single, frozen \u03b3, \u03bc": "Single, constant $\\gamma,\\mu$",
    "Null (no feedback, incremental)": "Null",
    "Single-sigmoid (+feedback, incremental)": "Single",
    "Double-sigmoid (incremental)": "Double",
    "Double, frozen \u03b3, \u03bc (incremental)": "Double, constant $\\gamma,\\mu$",
    "Single, frozen \u03b3, \u03bc (incremental)": "Single, constant $\\gamma,\\mu$",
}

MODEL_ORDER = ["Null", "Single", "Double", "Double, constant $\\gamma,\\mu$", "Single, constant $\\gamma,\\mu$"]


def load_and_relabel(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    df.index = [MODEL_LABELS.get(name, name) for name in df.index]
    return df.loc[MODEL_ORDER]


def build_table(basis_label, df, label_suffix):
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    lines.append(
        rf"    \caption{{Model comparison summary, {basis_label} basis. "
        r"SSE: sum of squared error. $k$: number of free parameters. "
        r"Lower AIC/AICc/BIC indicates a better balance of fit against complexity.}"
    )
    lines.append(r"    \begin{tabular}{lccccc}")
    lines.append(r"        \toprule")
    lines.append(r"        \textbf{Model} & \textbf{$k$} & \textbf{SSE} & \textbf{AIC} & \textbf{AICc} & \textbf{BIC} \\")
    lines.append(r"        \midrule")
    for name in MODEL_ORDER:
        row = df.loc[name]
        lines.append(
            f"        {name} & {int(row['k'])} & {row['SSE']:.1f} & "
            f"{row['AIC']:.2f} & {row['AICc']:.2f} & {row['BIC']:.2f} \\\\"
        )
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(rf"    \label{{tab:model_comparison_summary_{label_suffix}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    cum_df = load_and_relabel(CUM_CSV)
    incr_df = load_and_relabel(INCR_CSV)

    cum_table = build_table("cumulative", cum_df, "cum")
    incr_table = build_table("incremental (weekly)", incr_df, "incr")

    full_tex = cum_table + "\n\n" + incr_table
    print(full_tex)

    out_path = os.path.join(BASE_DIR, "model_comparison_summary_table.tex")
    with open(out_path, "w") as f:
        f.write(full_tex)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()