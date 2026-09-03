"""
vetlyanka_IWM_constant_mu_check.py

Recomputes the mu(t)-held-constant robustness check (IWM under the real
fitted mu(t) vs. mu held at its initial or epidemic-mean value) against
the CURRENT, FINAL, locked Vetlyanka double-sigmoid fit. The previous
numbers reported in the manuscript (63.7% -> 49.5%/58.4%) were computed
against an earlier, superseded fit and must not be reused.

Uses the exact mu_function(t, mu1, mu2, x0, c) from
plague_double_sigmoid_model.py (mu(t) and gamma(t) both share beta's
rise-timing parameters c, x0, per the deliberate DOppner-motivated
choice tying clinical-progression timing to the same underlying
bubonic-to-pneumonic transition driving transmission's own rise).

DATA LAYOUT: reads the fit from data/fits/double/.
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOUBLE_FIT_CSV = os.path.join(BASE_DIR, "data", "fits", "double", "vetlyanka_polished_fit.csv")

PARAM_NAMES = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2",
               "mu1", "mu2", "T_perceive", "sensitivity", "dispose_rate"]

t_start, t_end, dt = 0, 22, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]
alpha = 0.000641025641025641


def double_sigmoid(t, b1, b2, x0, x1, c, c1):
    increase = 1 / (1 + np.exp(-c * (t - x0)))
    decline = 1 / (1 + np.exp(c1 * (t - x1)))
    return b1 + (b2 - b1) * increase * decline


def mu_function(t, mu1, mu2, x0, c):
    return mu1 + (mu2 - mu1) / (1 + np.exp((-t + x0) * c))


def gamma_function(t, gamma1, gamma2, x0, c):
    return gamma1 + (gamma2 - gamma1) / (1 + np.exp((-t + x0) * c))


def plague_model_variant(t, y, params, mu_override=None):
    """Same as the real plague_model, but mu can be overridden to a
    fixed constant instead of computed from mu_function(t, ...)."""
    b1, b2, x0, x1, c, c1, gamma1, gamma2, mu1, mu2, T_perceive, sensitivity, dispose_rate = params
    S, I, R, DI, DDI, DR, P = y
    beta = double_sigmoid(t, b1, b2, x0, x1, c, c1)
    mu = mu_override if mu_override is not None else mu_function(t, mu1, mu2, x0, c)
    gamma = gamma_function(t, gamma1, gamma2, x0, c)
    dSdt = -S * (I + DI) / (S + I + R) * beta * np.exp(-sensitivity * P) - alpha * S
    dIdt = S * (I + DI) / (S + I + R) * beta * np.exp(-sensitivity * P) - alpha * I - mu * I - gamma * I
    dRdt = gamma * I - alpha * R
    dDIdt = mu * I - dispose_rate * DI
    dDDIdt = mu * I
    dDRdt = mu * I + alpha * (S + I + R)
    dPdt = (mu * I - P) / T_perceive
    return [dSdt, dIdt, dRdt, dDIdt, dDDIdt, dDRdt, dPdt]


def compute_IWM(params, mu_override=None):
    sol = solve_ivp(plague_model_variant, [t_start, t_end], initial_conditions,
                     args=(params, mu_override), t_eval=t_points, method="RK45")
    I = sol.y[1]
    P = sol.y[6]
    sensitivity = params[11]
    rho_beta = (1 - np.exp(-sensitivity * P)) * 100
    IWM = np.sum(rho_beta * I) / np.sum(I)
    return IWM


def main():
    df = pd.read_csv(DOUBLE_FIT_CSV).set_index("Parameter")
    params = np.array([df.loc[name, "Best_Fit"] for name in PARAM_NAMES])
    mu1, mu2, x0, c = params[8], params[9], params[2], params[4]

    mu_initial = mu_function(0, mu1, mu2, x0, c)
    mu_mean = np.mean([mu_function(t, mu1, mu2, x0, c) for t in t_points])

    print(f"Fitted mu1={mu1:.4f}, mu2={mu2:.4f}, x0={x0:.4f}, c={c:.4f}")
    print(f"mu(0) [initial value] = {mu_initial:.6f}")
    print(f"mean mu(t) over observation window = {mu_mean:.6f}")

    IWM_fitted = compute_IWM(params, mu_override=None)
    IWM_initial = compute_IWM(params, mu_override=mu_initial)
    IWM_mean = compute_IWM(params, mu_override=mu_mean)

    print(f"\nIWM, fitted (time-varying) mu(t):    {IWM_fitted:.1f}%")
    print(f"IWM, mu held at initial value:        {IWM_initial:.1f}%")
    print(f"IWM, mu held at epidemic-mean value:   {IWM_mean:.1f}%")
    print(f"\nManuscript sentence (Vetlyanka only, current fit):")
    print(f'"In Vetlyanka, IWM decreased from {IWM_fitted:.1f}% in the fitted model to '
          f'{IWM_initial:.1f}% under initial-value mu and {IWM_mean:.1f}% under mean-value mu."')


if __name__ == "__main__":
    main()