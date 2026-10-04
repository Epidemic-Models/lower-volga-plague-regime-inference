"""
vetlyanka_IWM_constant_mu_check.py   (step 06 -- run after 02_fits)

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

IWM = infection-weighted mean of the behavioural reduction in transmission,
rho_beta(t) = 100 * (1 - exp(-sensitivity * P(t))), weighted by I(t) over the
whole simulation (t = 0..22). It does not depend on the observation-time
alignment.

Reads: 02_fits/results/double/vetlyanka_polished_fit.csv
Writes: 06_robustness/results/IWM_constant_mu_check.txt
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))   # .../vetlyanka/06_robustness
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):  # walk up to vetlyanka/, which contains src/
    if os.path.dirname(ROOT) == ROOT:
        raise FileNotFoundError("Could not find the src/ folder above this script.")
    ROOT = os.path.dirname(ROOT)
DOUBLE_FIT_CSV = os.path.join(ROOT, "02_fits", "results", "double", "vetlyanka_polished_fit.csv")
OUTPUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    if not os.path.exists(DOUBLE_FIT_CSV):
        raise FileNotFoundError(f"{DOUBLE_FIT_CSV} not found -- run 02_fits/polish_vetlyanka_fit_double.py first.")
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
    direction = ("decreased" if max(IWM_initial, IWM_mean) < IWM_fitted
                 else "increased" if min(IWM_initial, IWM_mean) > IWM_fitted else "changed")
    sentence = (f'"In Vetlyanka, IWM {direction} from {IWM_fitted:.1f}% in the fitted model to '
                f'{IWM_initial:.1f}% under initial-value mu and {IWM_mean:.1f}% under mean-value mu."')
    print(f"\nManuscript sentence (Vetlyanka only, current fit):\n{sentence}")
    with open(os.path.join(OUTPUT_DIR, "IWM_constant_mu_check.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"mu1={mu1:.6f} mu2={mu2:.6f} x0={x0:.6f} c={c:.6f}\n"
                 f"mu(0)={mu_initial:.6f} mean_mu={mu_mean:.6f}\n"
                 f"IWM_fitted={IWM_fitted:.3f} IWM_initial={IWM_initial:.3f} IWM_mean={IWM_mean:.3f}\n{sentence}\n")
    print(f"\nSaved {os.path.join(OUTPUT_DIR, 'IWM_constant_mu_check.txt')}")


if __name__ == "__main__":
    main()