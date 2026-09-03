import pandas as pd
from vetlyanka_bounds import PARAMETERS, PARAM_NAMES_SINGLE, PARAM_NAMES_NULL
import os

T_PERCEIVE_FIXED_SINGLE = 3.409  # must match polish_vetlyanka_fit_single_freeze_mu_gamma_FIXED.py

# LaTeX names for the 13 parameters, in the same order as PARAMETERS
final_param_names = [
    '$\\beta_B$', '$\\beta_P$', '$x_0$', '$x_1$',
    '$\\kappa_{\\text{inc}}$', '$\\kappa_{\\text{dec}}$',
    '$\\gamma_1$', '$\\gamma_2$',
    '$\\mu_1$', '$\\mu_2$',
    '$\\tau_{\\text{P}}$', '$\\lambda_P$', '$\\xi$'
]

# Parameters that are in "per week" units and should be converted to days
DAYS_PARAMS = {'b1', 'b2', 'gamma1', 'gamma2', 'mu1', 'mu2', 'dispose_rate'}

# Single-sigmoid model does not have x1, c1
SINGLE_SIGMOID_PARAMS = set(PARAM_NAMES_SINGLE)  # x1, c1 excluded

# Null model has its own subset of parameters
NULL_PARAMS = set(PARAM_NAMES_NULL)


def weeks_to_days(value):
    return round(7 / value, 2) if value != 0 else "-"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

double_df = pd.read_csv(os.path.join(BASE_DIR, "data", "fits", "double", "vetlyanka_polished_fit.csv")).set_index('Parameter')
single_df = pd.read_csv(os.path.join(BASE_DIR, "data", "fits", "single", "vetlyanka_polished_fit_single.csv")).set_index('Parameter')
null_df = pd.read_csv(os.path.join(BASE_DIR, "data", "fits", "null", "vetlyanka_polished_fit_null.csv")).set_index('Parameter')
double_frozen_df = pd.read_csv(os.path.join(BASE_DIR, "data", "fits", "double_frozen", "vetlyanka_constant_mu_gamma_polished_fit.csv")).set_index('Parameter')
single_frozen_df = pd.read_csv(os.path.join(BASE_DIR, "data", "fits", "single_frozen", "vetlyanka_constant_mu_gamma_single_FIXED_fit.csv")).set_index('Parameter')
latex_table = "\\begin{table}[H]\n"
latex_table += "    \\centering\n"
latex_table += "    \\scriptsize \n"
latex_table += (
    "    \\caption{Parameter ranges and best-fit values for Vetlyanka region "
    "across five models (values and equivalent days for rate parameters). "
    "$^*$ = fixed by argument, not fitted. $^\\dagger$ = the double-frozen/"
    "single-frozen models collapse $\\gamma_1,\\gamma_2$ (resp. $\\mu_1,\\mu_2$) "
    "into one constant fit under the UNION of both individual ranges shown, "
    "not the single narrower range printed in this row -- the value may "
    "appear outside the row's own range while being valid under its real bound.}\n"
)
latex_table += (
    "    \\begin{tabular}{@{\\hskip 3pt}p{2.4cm}"
    "@{\\hskip 3pt}p{2.2cm}"  # range
    "@{\\hskip 3pt}p{1.8cm}@{\\hskip 1pt}p{1.4cm}"  # Null, days
    "@{\\hskip 3pt}p{1.8cm}@{\\hskip 1pt}p{1.4cm}"  # Single, days
    "@{\\hskip 3pt}p{1.8cm}@{\\hskip 1pt}p{1.4cm}"  # Double, days
    "@{\\hskip 3pt}p{1.8cm}@{\\hskip 1pt}p{1.4cm}"  # Double-frozen, days
    "@{\\hskip 3pt}p{1.8cm}@{\\hskip 1pt}p{1.4cm}}" # Single-frozen, days
    "\n"
)
latex_table += "        \\toprule\n"
latex_table += (
    "        \\textbf{Parameter} & \\textbf{Range} "
    "& \\textbf{Null} & \\textbf{(days)} "
    "& \\textbf{Single} & \\textbf{(days)} "
    "& \\textbf{Double} & \\textbf{(days)} "
    "& \\textbf{Double-frozen} & \\textbf{(days)} "
    "& \\textbf{Single-frozen} & \\textbf{(days)} \\\\\n"
)
latex_table += "        \\midrule\n"


def get_double_frozen_value(param):
    """Map PARAM_NAMES to the reduced double-frozen fit (gamma_const, mu_const, etc.)."""
    if param == 'gamma1' or param == 'gamma2':
        return double_frozen_df.loc['gamma_const', 'Best_Fit'], False
    if param == 'mu1' or param == 'mu2':
        return double_frozen_df.loc['mu_const', 'Best_Fit'], False
    if param in double_frozen_df.index:
        return double_frozen_df.loc[param, 'Best_Fit'], False
    return None, False  # not present


def get_single_frozen_value(param):
    """
    Map PARAM_NAMES to the reduced single-frozen fit (gamma_const, mu_const, etc.).

    Returns (value, is_fixed). T_perceive is FIXED at T_PERCEIVE_FIXED_SINGLE in
    this model, not fitted -- it is genuinely part of the model (unlike x1/c1,
    which single-sigmoid structurally doesn't have at all), so it must show its
    real value in the table, not "--". is_fixed=True marks it for the asterisk.

    gamma1 and gamma2 both collapse into a single gamma_const value -- there is
    no separate 'gamma1' row in this CSV; gamma1 must be mapped to gamma_const
    exactly like gamma2, not read as its own row.
    """
    if param == 'gamma1' or param == 'gamma2':
        return single_frozen_df.loc['gamma_const', 'Best_Fit'], False
    if param == 'mu1' or param == 'mu2':
        return single_frozen_df.loc['mu_const', 'Best_Fit'], False
    if param == 'T_perceive':
        return T_PERCEIVE_FIXED_SINGLE, True
    if param in single_frozen_df.index:
        return single_frozen_df.loc[param, 'Best_Fit'], False
    return None, False  # not present (x1, c1)


T_PERCEIVE_FIXED_SIGMOID = 3.571  # must match polish_vetlyanka_fit_single_FIXED.py -- T_perceive
                                    # is fixed, not fitted, in the current single-sigmoid model;
                                    # not a row in vetlyanka_polished_fit_single.csv anymore

for idx, (param, min_range, max_range) in enumerate(PARAMETERS):
    # Double
    double_fit = double_df.loc[param, 'Best_Fit']
    double_days = weeks_to_days(double_fit) if param in DAYS_PARAMS else "-"

    # Single
    if param == 'T_perceive':
        single_fit_str = f"{T_PERCEIVE_FIXED_SIGMOID}$^*$"
        single_days = weeks_to_days(T_PERCEIVE_FIXED_SIGMOID) if param in DAYS_PARAMS else "-"
    elif param in SINGLE_SIGMOID_PARAMS:
        single_fit = single_df.loc[param, 'Best_Fit']
        single_fit_str = f"{single_fit}"
        single_days = weeks_to_days(single_fit) if param in DAYS_PARAMS else "-"
    else:
        single_fit_str = "--"
        single_days = "--"

    # Null
    if param in NULL_PARAMS:
        null_fit = null_df.loc[param, 'Best_Fit']
        null_fit_str = f"{null_fit}"
        null_days = weeks_to_days(null_fit) if param in DAYS_PARAMS else "-"
    else:
        null_fit_str = "--"
        null_days = "--"

    # Double-frozen
    df_frozen_val, df_frozen_fixed = get_double_frozen_value(param)
    if df_frozen_val is not None:
        marker = "$^*$" if df_frozen_fixed else ""
        # Flag with a dagger if this value's REAL bound (the span, for the
        # collapsed gamma_const/mu_const rows) differs from the row's own
        # printed [min_range, max_range] -- that range is gamma1's/gamma2's/
        # mu1's/mu2's ORIGINAL individual bound, not what gamma_const/mu_const
        # was actually fit under, so a value can look out-of-range here while
        # being perfectly valid under its real, wider span bound.
        span_flag = "$^\\dagger$" if param in {"gamma1", "gamma2", "mu1", "mu2"} else ""
        double_frozen_str = f"{df_frozen_val}{marker}{span_flag}"
        double_frozen_days = weeks_to_days(df_frozen_val) if param in DAYS_PARAMS else "-"
    else:
        double_frozen_str = "--"
        double_frozen_days = "--"

    # Single-frozen
    sf_frozen_val, sf_frozen_fixed = get_single_frozen_value(param)
    if sf_frozen_val is not None:
        marker = "$^*$" if sf_frozen_fixed else ""
        span_flag = "$^\\dagger$" if param in {"gamma1", "gamma2", "mu1", "mu2"} else ""
        single_frozen_str = f"{sf_frozen_val}{marker}{span_flag}"
        single_frozen_days = weeks_to_days(sf_frozen_val) if param in DAYS_PARAMS else "-"
    else:
        single_frozen_str = "--"
        single_frozen_days = "--"

    latex_table += (
        f"        {final_param_names[idx]} & [{min_range}, {max_range}] "
        f"& {null_fit_str} & {null_days} "
        f"& {single_fit_str} & {single_days} "
        f"& {double_fit} & {double_days} "
        f"& {double_frozen_str} & {double_frozen_days} "
        f"& {single_frozen_str} & {single_frozen_days} \\\\\n"
    )

latex_table += "        \\bottomrule\n"
latex_table += "    \\end{tabular}\n"
latex_table += "    \\label{tab:vetlyanka_ranges_best_fit_days_five_models}\n"
latex_table += "\\end{table}"

print(latex_table)