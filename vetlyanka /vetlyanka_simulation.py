# vetlyanka_simulation.py

import numpy as np
from scipy.integrate import solve_ivp

from plague_double_sigmoid_model import plague_model


# Time settings and initial conditions: match your fitting scripts
t_start = 0.0
t_end = 22.0
dt = 0.01

t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
n_weeks = 22

# S0, I0, R0, DI0, DDI0, DR0, P0
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]


def simulate_vetlyanka(params):
    """
    Run the double-sigmoid Vetlyanka model and return the 22
    weekly cumulative death counts (DR), matching your observed data.

    Parameters
    ----------
    params : array-like, shape (13,)
        Parameter vector in the same order as PARAM_NAMES in
        vetlyanka_bounds.py:
            [b1, b2, x0, x1, c, c1,
             gamma1, gamma2, mu1, mu2,
             T_perceive, sensitivity, dispose_rate]

    Returns
    -------
    dr_weekly : ndarray, shape (22,)
        Weekly cumulative deaths DR(t) at weeks 1..22, or
        None if the ODE solve fails.
    """
    params = np.asarray(params, dtype=float)

    if params.shape != (13,):
        raise ValueError(
            f"simulate_vetlyanka expected 13 parameters, "
            f"got shape {params.shape}."
        )

    try:
        sol = solve_ivp(
            plague_model,
            [t_start, t_end],
            initial_conditions,
            args=(params,),
            t_eval=t_points,
            method="RK45",
        )
    except Exception:
        return None

    if not sol.success:
        return None

    if not np.all(np.isfinite(sol.y)):
        return None

    # DR is y[5]; sample once per week (every 1/dt steps)
    dr_weekly = sol.y[5][::steps_per_week][:n_weeks]

    if dr_weekly.shape[0] != n_weeks:
        return None

    return dr_weekly
