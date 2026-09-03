"""
main_text_parameter_table.py

Generates the minimal, main-text parameter table (Option B): double-
sigmoid vs single-sigmoid-with-feedback, time-varying gamma/mu only,
restricted to the 7 parameters the results text actually discusses by
value (b1, x1, gamma1, gamma2, mu1, mu2, T_perceive). The full five-
model, all-parameter table (five_model_parameter_table_FIXED.py)
belongs in SI.

T_perceive: the single-sigmoid fit FIXES T_perceive at 3.571 rather
than fitting it -- not a row in vetlyanka_polished_fit_single.csv,
supplied here as the same fixed constant used in the real fit.

x1 has no single-regime value at all (single-sigmoid has no second
transition) -- printed as "--", a deliberate, visible piece of evidence
for the argument, not an omission.

DATA LAYOUT: reads both fits from data/fits/.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOUBLE_FIT_CSV = os.path.join(BASE_DIR, "data", "fits", "double", "vetlyanka_polished_fit.csv")
SINGLE_FIT_CSV = os.path.join(BASE_DIR, "data", "fits", "single", "vetlyanka_polished_fit_single.csv")

T_PERCEIVE_FIXED_SIGMOID = 3.571  # must match polish_vetlyanka_fit_single_FIXED.py

# (row label, double CSV param name, single CSV param name or None, has_days_column)
# Row labels use the paper's OFFICIAL notation (matching
# tab:model_parameters in Methods), not the internal Python variable
# names (b1, gamma1, etc.) -- beta_B/beta_P, gamma_B/gamma_P, mu_B/mu_P
# all use letter subscripts (Base/Peak), not numeric ones.
ROWS = [
    (r"$\beta_B$ (transmission rate, lower bound)", "b1", "b1", True),
    (r"$x_1$ (decline onset, wk)", "x1", None, False),
    (r"$\gamma_B$ (recovery rate, lower bound)", "gamma1", "gamma1", True),
    (r"$\gamma_P$ (recovery rate, upper bound)", "gamma2", "gamma2", True),
    (r"$\mu_B$ (death rate, lower bound)", "mu1", "mu1", True),
    (r"$\mu_P$ (death rate, upper bound)", "mu2", "mu2", True),
    (r"$\tau_P$ (time to perceive risk, wk)", "T_perceive", "T_PERCEIVE_FIXED", False),
]


def weeks_to_days(value):
    return f"{7 / value:.1f}" if value != 0 else "-"


def main():
    double_df = pd.read_csv(DOUBLE_FIT_CSV).set_index("Parameter")
    single_df = pd.read_csv(SINGLE_FIT_CSV).set_index("Parameter")  # no T_perceive row -- fixed, see above

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    lines.append(
        r"    \caption{Best-fit parameters for the double-sigmoid and single-sigmoid-with-feedback "
        r"models (time-varying $\gamma,\mu$), with equivalent days for rate parameters. "
        r"$^*$ = fixed by argument, not fitted (see Methods).}"
    )
    lines.append(
        r"    \begin{tabular}{@{\hskip 3pt}p{2.4cm}@{\hskip 6pt}p{1.6cm}@{\hskip 1pt}p{1.3cm}"
        r"@{\hskip 10pt}p{1.6cm}@{\hskip 1pt}p{1.3cm}}"
    )
    lines.append(r"        \toprule")
    lines.append(r"        \textbf{Parameter} & \textbf{Double} & \textbf{(days)} & \textbf{Single} & \textbf{(days)} \\")
    lines.append(r"        \midrule")

    for label, double_key, single_key, has_days in ROWS:
        double_val = double_df.loc[double_key, "Best_Fit"]
        double_str = f"{double_val:.2f}" if double_key != "x1" else f"{double_val:.2f}"
        double_days = weeks_to_days(double_val) if has_days else "--"

        if single_key is None:
            single_str = "--"
            single_days = "--"
        elif single_key == "T_PERCEIVE_FIXED":
            single_str = f"{T_PERCEIVE_FIXED_SIGMOID}$^*$"
            single_days = "--"
        else:
            single_val = single_df.loc[single_key, "Best_Fit"]
            single_str = f"{single_val:.2f}" if single_key != "x1" else f"{single_val:.2f}"
            single_days = weeks_to_days(single_val) if has_days else "--"

        lines.append(f"        {label} & {double_str} & {double_days} & {single_str} & {single_days} \\\\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \label{tab:main_params}")
    lines.append(r"\end{table}")

    table_tex = "\n".join(lines)
    print(table_tex)

    out_path = os.path.join(BASE_DIR, "main_text_parameter_table.tex")
    with open(out_path, "w") as f:
        f.write(table_tex)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()