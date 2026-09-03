"""
check_rebound_not_numerical_artefact.py

Directly tests whether the single-sigmoid-with-feedback model's predicted
second wave (see plot_extended_projection_v2.py) is a genuine feature of
the fitted ODE system, or a numerical-integration artefact -- e.g. I(t)
being driven to a numerically tiny but biologically meaningless residual
during the trough, which then spuriously regrows due to floating-point
noise rather than real dynamics.

Two independent checks, both against the same fitted parameters:
  1. Is I(t) at its lowest point in the trough (weeks 18-25) a genuinely
     meaningful number, or a near-zero numerical residue?
  2. Does the rebound's timing and magnitude change if the ODE is solved
     with much tighter error tolerances, or with a completely different
     integration algorithm (Radau, an implicit method suited to stiff
     systems)? If the result is a numerical artefact, tightening
     precision or switching solver families should change or eliminate
     it. If it's the genuine solution, all three approaches should agree.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from plague_single_sigmoid_model import plague_model
from vetlyanka_bounds import PARAM_NAMES, PARAM_NAMES_SINGLE

X1_C1_PLACEHOLDER = 1.0
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]  # S0, I0, R0, DI0, DDI0, DR0, P0
t_start, t_end_extended = 0, 52
t_eval = np.arange(t_start, t_end_extended + 0.01, 0.01)


def load_best_fit(csv_path: str, param_order: list) -> np.ndarray:
    df = pd.read_csv(csv_path).set_index("Parameter")
    return np.array([df.loc[name, "Best_Fit"] for name in param_order])


def expand_single_to_full(single_params: np.ndarray) -> np.ndarray:
    single_dict = dict(zip(PARAM_NAMES_SINGLE, single_params))
    return np.array([single_dict.get(name, X1_C1_PLACEHOLDER) for name in PARAM_NAMES])


single_best = expand_single_to_full(
    load_best_fit("vetlyanka_polished_fit_single.csv", PARAM_NAMES_SINGLE)
)

runs = [
    ("Default RK45 (loose tolerances)",
     dict(method="RK45")),  # scipy defaults: rtol=1e-3, atol=1e-6
    ("RK45, tolerances tightened 10,000x",
     dict(method="RK45", rtol=1e-10, atol=1e-12)),
    ("Radau (implicit stiff-solver), same tight tolerances",
     dict(method="Radau", rtol=1e-10, atol=1e-12)),
]

print(f"{'Method':<55} {'min I, wk18-25':>16} {'wk22':>8} {'wk25':>8} {'wk30':>8}")
for label, solver_kwargs in runs:
    sol = solve_ivp(plague_model, [t_start, t_end_extended], initial_conditions,
                     args=(single_best,), t_eval=t_eval, **solver_kwargs)

    I = sol.y[1]
    t = sol.t
    trough_mask = (t >= 18) & (t <= 25)
    I_trough_min = I[trough_mask].min()

    DR = sol.y[5]
    weekly_DR = DR[::100][:52]
    weekly_deaths = np.diff(np.concatenate(([0.0], weekly_DR)))

    print(f"{label:<55} {I_trough_min:>16.6f} {weekly_deaths[22]:>8.3f} "
          f"{weekly_deaths[25]:>8.3f} {weekly_deaths[30]:>8.3f}")

print("\nInterpretation:")
print("- If these three rows agree closely, the rebound is the genuine solution")
print("  of the fitted ODE system, not a numerical-precision artefact.")
print("- I in the trough should also be checked against 0: a value many orders")
print("  of magnitude below 1 (e.g. 1e-10) would indicate a numerically")
print("  meaningless residual; a value of order 0.1-1 (comparable to the")
print("  model's own I0=1 starting seed) indicates a real, if small, level")
print("  of ongoing transmission -- not noise.")
