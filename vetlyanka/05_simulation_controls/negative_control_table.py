"""
negative_control_table.py   (step 05 -- run after vetlyanka_negative_control_recovery.py)

Builds the SI LaTeX table for the five single-regime negative-control
experiments. Each synthetic dataset was generated from the fitted
single-regime model with Poisson noise on weekly deaths; both models were
refitted with the full multi-start procedure (single-regime: T_perceive fixed
at the value fitted by the double-regime model on the same dataset, k = 10).

Columns per replicate: SSE single, SSE double, Delta AIC, Delta AICc,
Delta BIC, fitted x1 of the double-regime model (marked when at its upper bound).

Sign convention (as in the published table):
    Delta IC = IC_double - IC_single   -> positive values favour the single-regime model.

Reads:  05_simulation_controls/results/negative_control/vetlyanka_negative_control_results.csv
Writes: 05_simulation_controls/results/negative_control/negative_control_table.tex
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "negative_control")
RESULTS_CSV = os.path.join(RESULTS_DIR, "vetlyanka_negative_control_results.csv")
OUTPUT_TEX = os.path.join(RESULTS_DIR, "negative_control_table.tex")

import pandas as pd


def main():
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"{RESULTS_CSV} not found -- run vetlyanka_negative_control_recovery.py first.")
    df = pd.read_csv(RESULTS_CSV)

    # Delta IC = IC_double - IC_single (positive favours single)
    df["dAIC"] = df["aic_double"] - df["aic_single"]
    df["dAICc"] = df["aicc_double"] - df["aicc_single"]
    df["dBIC"] = df["bic_double"] - df["bic_single"]

    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \small",
        r"    \caption{Negative-control recovery experiment using synthetic single-regime data. "
        r"Five datasets were generated from the fitted single-regime model with Poisson observation "
        r"noise on weekly deaths, and both models were refitted to each dataset with the same "
        r"multi-start procedure as for the observed Vetlyanka data (single-regime model: "
        r"$T_{\mathrm{perceive}}$ fixed at the value fitted by the double-regime model on the same "
        r"dataset, $k=10$; double-regime model: $k=13$). "
        r"$\Delta\mathrm{IC}=\mathrm{IC}_{\mathrm{double}}-\mathrm{IC}_{\mathrm{single}}$, so positive "
        r"values favour the single-regime model. The fitted $x_1$ refers to the double-regime fit; "
        r"the generating process has no decline onset. $^{\dagger}$At the upper bound of $x_1$ (20 weeks).}",
        r"    \begin{tabular}{crrrrrr}",
        r"        \toprule",
        r"        \textbf{Rep.} & \textbf{SSE single} & \textbf{SSE double} & "
        r"$\boldsymbol{\Delta}$\textbf{AIC} & $\boldsymbol{\Delta}$\textbf{AICc} & "
        r"$\boldsymbol{\Delta}$\textbf{BIC} & \textbf{Fitted $x_1$ (wk)} \\",
        r"        \midrule",
    ]
    for _, row in df.iterrows():
        mark = r"$^{\dagger}$" if bool(row.get("x1_at_upper_bound", False)) else ""
        lines.append(f"        {int(row['replicate'])} & {row['sse_single']:.2f} & {row['sse_double']:.2f} & "
                     f"{row['dAIC']:.2f} & {row['dAICc']:.2f} & {row['dBIC']:.2f} & "
                     f"{row['x1_fitted']:.2f}{mark} \\\\")
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \label{tab:negative_control_recovery}",
        r"\end{table}",
    ]
    table_tex = "\n".join(lines)
    print(table_tex)
    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(table_tex)
    print(f"\nSaved {OUTPUT_TEX}")


if __name__ == "__main__":
    main()