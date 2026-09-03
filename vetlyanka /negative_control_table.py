"""
negative_control_table.py

Generates a Supporting Information LaTeX table summarizing the five
single-regime negative-control recovery experiments for Vetlyanka.

Each synthetic dataset was generated from the fitted single-regime model
using Poisson observation noise on weekly incremental deaths, after which
both the single- and double-regime models were refitted using the same
multi-start fitting procedure.

The table reports, for each replicate:
    - SSE for the fitted single-regime model
    - SSE for the fitted double-regime model
    - Delta AIC
    - Delta AICc
    - Delta BIC
    - fitted decline-onset parameter x1 from the double-regime fit

All information-criterion differences are defined as

    Delta IC = IC_double - IC_single

so positive values favor the single-regime model.

IMPORTANT:
The CSV used here should come from the corrected negative-control script
in which the single-regime model matches the main analysis exactly,
including T_perceive fixed at 3.571 and k = 10 rather than treating it
as a fitted parameter.

DATA LAYOUT:
    reads:
        data/diagnostics/vetlyanka_negative_control_results.csv

    writes:
        negative_control_table.tex
"""

import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_CSV = os.path.join(
    BASE_DIR,
    "data",
    "diagnostics",
    "vetlyanka_negative_control_results.csv",
)

OUTPUT_TEX = os.path.join(
    BASE_DIR,
    "negative_control_table.tex",
)


def main():
    df = pd.read_csv(RESULTS_CSV)

    # ------------------------------------------------------------
    # Calculate information-criterion differences.
    #
    # Convention:
    #     Delta IC = IC_double - IC_single
    #
    # Positive values therefore favor the single-regime model.
    # ------------------------------------------------------------
    df["delta_AIC"] = df["aic_double"] - df["aic_single"]
    df["delta_AICc"] = df["aicc_double"] - df["aicc_single"]
    df["delta_BIC"] = df["bic_double"] - df["bic_single"]

    lines = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    lines.append(
        r"    \caption{Negative-control recovery experiment using synthetic "
        r"single-regime data. Five datasets were generated from the fitted "
        r"single-regime model using Poisson observation noise on weekly "
        r"incremental deaths, and both competing models were independently "
        r"refitted to each dataset using the same multi-start procedure as "
        r"for the observed Vetlyanka data. Information-criterion differences "
        r"are defined as $\Delta\mathrm{IC}=\mathrm{IC}_{\mathrm{double}}"
        r"-\mathrm{IC}_{\mathrm{single}}$, so positive values favor the "
        r"single-regime model. The fitted $x_1$ value refers to the "
        r"double-regime fit; the true generating process contains no "
        r"decline-onset parameter.}"
    )

    lines.append(
        r"    \begin{tabular}{c"
        r"rrrrrr}"
    )

    lines.append(r"        \toprule")
    lines.append(
        r"        \textbf{Rep.} & "
        r"\textbf{SSE single} & "
        r"\textbf{SSE double} & "
        r"$\boldsymbol{\Delta}$\textbf{AIC} & "
        r"$\boldsymbol{\Delta}$\textbf{AICc} & "
        r"$\boldsymbol{\Delta}$\textbf{BIC} & "
        r"\textbf{Fitted $x_1$ (wk)} \\"
    )
    lines.append(r"        \midrule")

    for _, row in df.iterrows():
        lines.append(
            "        "
            f"{int(row['replicate'])} & "
            f"{row['sse_single']:.2f} & "
            f"{row['sse_double']:.2f} & "
            f"{row['delta_AIC']:.2f} & "
            f"{row['delta_AICc']:.2f} & "
            f"{row['delta_BIC']:.2f} & "
            f"{row['x1_fitted']:.2f} \\\\"
        )

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \label{tab:negative_control_recovery}")
    lines.append(r"\end{table}")

    table_tex = "\n".join(lines)

    print(table_tex)

    with open(OUTPUT_TEX, "w") as f:
        f.write(table_tex)

    print(f"\nSaved {OUTPUT_TEX}")


if __name__ == "__main__":
    main()