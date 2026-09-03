# plague_model.py

import numpy as np

# --- Helper Functions ---

def mu_function(t, mu1, mu2, x0, c):
    """
    Time-dependent mortality rate using a sigmoid function.
    """
    return mu1 + (mu2 - mu1) / (1 + np.exp((-t + x0) * c))


def gamma_function(t, gamma1, gamma2, x0, c):
    """
    Time-dependent recovery rate using a sigmoid function.
    """
    return gamma1 + (gamma2 - gamma1) / (1 + np.exp((-t + x0) * c))


def double_sigmoid(t, b1, b2, x0, x1, c, c1):
    increase = 1 / (1 + np.exp(-c * (t - x0)))
    return b1 + (b2 - b1) * increase


# --- Main Plague Model ---

def plague_model(t, y, params):
    """
    System of ODEs modeling the spread of plague with delayed perception of death.

    Parameters:
    - t: Time
    - y: List of current state variables [S, I, R, DI, DDI, DR, P]
    - params: List of parameters (in order)

    Returns:
    - dydt: List of time derivatives for the system
    """
    # Constant external death rate
    alpha = 0.000641025641025641

    # Unpack parameters
    b1, b2, x0, x1, c, c1, gamma1, gamma2, mu1, mu2,  T_perceive, sensitivity, dispose_rate  = params

    # Unpack state variables
    S, I, R, DI, DDI, DR, P = y

    # Time-dependent parameter values
    beta = double_sigmoid(t, b1, b2, x0, x1, c, c1)
    mu = mu_function(t, mu1, mu2, x0, c)
    gamma = gamma_function(t, gamma1, gamma2, x0, c)

    # ODEs
    dSdt = -S * (I + DI) / (S + I + R) * beta * np.exp(-sensitivity * P) - alpha * S
    dIdt = S * (I + DI) / (S + I + R) * beta * np.exp(-sensitivity * P) - alpha * I - mu * I - gamma * I
    dRdt = gamma * I - alpha * R
    dDIdt = mu * I - dispose_rate * DI
    dDDIdt = mu * I
    dDRdt = mu * I + alpha * (S + I + R)
    dPdt = (mu * I - P) / T_perceive

    return [dSdt, dIdt, dRdt, dDIdt, dDDIdt, dDRdt, dPdt]