"""
five_model_parameter_table_FIXED.py   (step 08 -- run after all five fits in 02_fits)

Builds SI Table S2: parameter bounds and best-fit values for the standalone
Vetlyanka analysis across all five models, in the layout used in the SI
(paper symbols, rounded values, * = fixed, dagger = constant-rate parameter).

Everything is READ from the fit outputs in 02_fits/results/, including the
fixed T_perceive values (stored as "fixed" rows in the single and single-frozen
fit CSVs) -- nothing is typed in by hand.

Writes: 08_figures_tables/results/five_model_parameter_table.tex
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # .../vetlyanka/08_figures_tables
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):  # walk up to vetlyanka/, which contains src/
    if os.path.dirname(ROOT) == ROOT:
        raise FileNotFoundError("Could not find the src/ folder above this script.")
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd
from vetlyanka_bounds import PARAMETERS

FITS = os.path.join(ROOT, "02_fits", "results")
OUTPUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_FILES = [   # column order of the table
    ("Null", "null", "vetlyanka_polished_fit_null.csv"),
    ("Single", "single", "vetlyanka_polished_fit_single.csv"),
    ("Double", "double", "vetlyanka_polished_fit.csv"),
    ("Double, const.", "double_frozen", "vetlyanka_constant_mu_gamma_polished_fit.csv"),
    ("Single, const.", "single_frozen", "vetlyanka_constant_mu_gamma_single_FIXED_fit.csv"),
]
SYMBOL = {"b1": r"$\beta_B$", "b2": r"$\beta_P$", "x0": r"$x_0$", "x1": r"$x_1$",
          "c": r"$\kappa_{\mathrm{inc}}$", "c1": r"$\kappa_{\mathrm{dec}}$",
          "gamma1": r"$\gamma_B$", "gamma2": r"$\gamma_P$", "mu1": r"$\mu_B$", "mu2": r"$\mu_P$",
          "T_perceive": r"$\tau_P$", "sensitivity": r"$\lambda_P$", "dispose_rate": r"$\xi$"}
CONST_MAP = {"gamma1": "gamma_const", "gamma2": "gamma_const", "mu1": "mu_const", "mu2": "mu_const"}


def load(subdir, filename):
    path = os.path.join(FITS, subdir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run the matching 02_fits script first.")
    df = pd.read_csv(path)
    status = df["Status"] if "Status" in df.columns else pd.Series(["fitted"] * len(df))
    return {p: (float(v), s == "fixed") for p, v, s in zip(df["Parameter"], df["Best_Fit"], status)}


def fmt(v):
    return f"{v:.3f}" if abs(v) < 0.1 else f"{v:.2f}"


def cell(fit, param):
    if param in CONST_MAP and CONST_MAP[param] in fit:
        v, fixed = fit[CONST_MAP[param]]
        return fmt(v) + r"$^{\dagger}$"
    if param not in fit:
        return "--"
    v, fixed = fit[param]
    if fixed and param == "sensitivity" and v == 0.0:     # null model: feedback switched off
        return "--"
    return fmt(v) + (r"$^{*}$" if fixed else "")


def main():
    fits = [(label, load(sub, f)) for label, sub, f in MODEL_FILES]
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \small",
        r"    \caption{Parameter ranges and best-fit values for the standalone Vetlyanka analysis across "
        r"all five models. This table provides the full parameter estimates underlying Table 1 in the "
        r"main text. $^{*}$Parameter fixed by model specification rather than estimated. $^{\dagger}$In "
        r"the constant-$\gamma,\mu$ models, $\gamma_B$ and $\gamma_P$ are replaced by a single constant "
        r"recovery rate, and $\mu_B$ and $\mu_P$ by a single constant mortality rate. Each constant "
        r"parameter was fitted over the union of the corresponding individual parameter ranges. "
        r"Consequently, its fitted value may fall outside the narrower range displayed for either "
        r"individual row while remaining within the bounds used for optimization.}",
        r"    \begin{tabular}{llccccc}",
        r"        \toprule",
        r"        \textbf{Parameter} & \textbf{Range} & " + " & ".join(rf"\textbf{{{l}}}" for l, _ in fits) + r" \\",
        r"        \midrule",
    ]
    for param, lo, hi in PARAMETERS:
        row = [SYMBOL.get(param, param), f"[{lo:g}, {hi:g}]"] + [cell(fit, param) for _, fit in fits]
        lines.append("        " + " & ".join(row) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}",
              r"    \label{tab:vetlyanka_ranges_best_fit_days_five_models}", r"\end{table}"]
    tex = "\n".join(lines)
    print(tex)
    out = os.path.join(OUTPUT_DIR, "five_model_parameter_table.tex")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(tex + "\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()