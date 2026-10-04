"""
main_text_parameter_table.py   (figures/tables step -- run after 02_fits)

Main-text Table 1: double-sigmoid vs single-sigmoid-with-feedback
(time-varying gamma, mu), restricted to the parameters the Results discuss by
value (beta_B, x1, gamma_B, gamma_P, mu_B, mu_P, tau_P), with equivalent days
(7 / weekly rate) for the rate parameters. The full five-model table is
five_model_parameter_table_FIXED.py (SI Table S2).

The single model's fixed tau_P is READ from its fit CSV (the "fixed" row), not
typed in. x1 has no single-regime value -- printed as "--" on purpose.

Writes: <this folder>/results/main_text_parameter_table.tex
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):  # walk up to vetlyanka/, which contains src/
    if os.path.dirname(ROOT) == ROOT:
        raise FileNotFoundError("Could not find the src/ folder above this script.")
    ROOT = os.path.dirname(ROOT)

import pandas as pd

FITS = os.path.join(ROOT, "02_fits", "results")
DOUBLE_FIT_CSV = os.path.join(FITS, "double", "vetlyanka_polished_fit.csv")
SINGLE_FIT_CSV = os.path.join(FITS, "single", "vetlyanka_polished_fit_single.csv")
OUTPUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# (row label, parameter name, show single value?, has days column)
ROWS = [
    (r"$\beta_B$ (transmission rate, lower bound)", "b1", True, True),
    (r"$x_1$ (decline onset, wk)", "x1", False, False),
    (r"$\gamma_B$ (recovery rate, lower bound)", "gamma1", True, True),
    (r"$\gamma_P$ (recovery rate, upper bound)", "gamma2", True, True),
    (r"$\mu_B$ (death rate, lower bound)", "mu1", True, True),
    (r"$\mu_P$ (death rate, upper bound)", "mu2", True, True),
    (r"$\tau_P$ (time to perceive risk, wk)", "T_perceive", True, False),
]


def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run the matching 02_fits script first.")
    df = pd.read_csv(path)
    status = df["Status"] if "Status" in df.columns else pd.Series(["fitted"] * len(df))
    return {p: (float(v), s == "fixed") for p, v, s in zip(df["Parameter"], df["Best_Fit"], status)}


def days(v):
    return f"{7 / v:.1f}" if v != 0 else "--"


def main():
    double, single = load(DOUBLE_FIT_CSV), load(SINGLE_FIT_CSV)
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \small",
        r"    \caption{Best-fit parameters for the double-sigmoid and single-sigmoid-with-feedback "
        r"models (time-varying $\gamma,\mu$), with equivalent days for rate parameters "
        r"(wk = weeks). $^*$ = fixed by model specification, not fitted (see Materials and methods).}",
        r"    \begin{tabular}{@{\hskip 3pt}p{2.4cm}@{\hskip 6pt}p{1.6cm}@{\hskip 1pt}p{1.3cm}"
        r"@{\hskip 10pt}p{1.6cm}@{\hskip 1pt}p{1.3cm}}",
        r"        \toprule",
        r"        \textbf{Parameter} & \textbf{Double} & \textbf{(days)} & \textbf{Single} & \textbf{(days)} \\",
        r"        \midrule",
    ]
    for label, key, show_single, has_days in ROWS:
        dv, _ = double[key]
        d_str, d_days = f"{dv:.2f}", (days(dv) if has_days else "--")
        if show_single and key in single:
            sv, fixed = single[key]
            s_str = f"{sv:.2f}" + (r"$^*$" if fixed else "")
            s_days = days(sv) if has_days else "--"
        else:
            s_str, s_days = "--", "--"
        lines.append(f"        {label} & {d_str} & {d_days} & {s_str} & {s_days} \\\\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"    \label{tab:main_params}", r"\end{table}"]
    tex = "\n".join(lines)
    print(tex)
    out = os.path.join(OUTPUT_DIR, "main_text_parameter_table.tex")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(tex + "\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()