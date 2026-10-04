# vetlyanka_simulation.py   (src/ -- shared helper)
"""
Run the double-sigmoid Vetlyanka model and return the 22 weekly cumulative
death counts (DR) at the observation times.

OBSERVATION TIMES: the observed value for week k is cumulative deaths at the
END of week k, i.e. model time t = k, so the model is sampled at t = 1..22.
(LEGACY_ALIGNMENT = True reproduces the published, off-by-one t = 0..21.)
"""

import numpy as np
from scipy.integrate import solve_ivp

from plague_double_sigmoid_model import plague_model

LEGACY_ALIGNMENT = False

t_start, t_end, dt = 0.0, 22.0, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
n_weeks = 22
OBS = slice(0, n_weeks) if LEGACY_ALIGNMENT else slice(1, n_weeks + 1)

initial_conditions = [1699, 1, 0, 0, 0, 0, 0]   # S0, I0, R0, DI0, DDI0, DR0, P0


def simulate_vetlyanka(params):
    """
    params: 13 values in PARAM_NAMES order
        [b1, b2, x0, x1, c, c1, gamma1, gamma2, mu1, mu2, T_perceive, sensitivity, dispose_rate]
    Returns the 22 weekly cumulative deaths at the observation times, or None if the solve fails.
    """
    params = np.asarray(params, dtype=float)
    if params.shape != (13,):
        raise ValueError(f"simulate_vetlyanka expected 13 parameters, got shape {params.shape}.")
    try:
        sol = solve_ivp(plague_model, [t_start, t_end], initial_conditions,
                        args=(params,), t_eval=t_points, method="RK45")
    except Exception:
        return None
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    dr_weekly = sol.y[5][::steps_per_week][OBS]
    return dr_weekly if dr_weekly.shape[0] == n_weeks else None