"""
single_frozen_T_perceive_sensitivity_check.py   (step 06 -- run after 02_fits)

Sensitivity of the constant-gamma/mu comparison to the FIXED T_perceive value
used in the single-frozen model.

The real single-frozen fit fixes T_perceive at the double-frozen fit's value
(when searched freely it runs to its ceiling together with dispose_rate at its
floor). This script refits single-frozen with T_perceive fixed at each value in
T_PERCEIVE_TEST_VALUES -- from very short lags (0.1-0.5, the region where the
Malta analysis found a crossover) to long lags -- plus the value actually used
in the paper, and asks whether single-frozen ever beats double-frozen.

For each value: LHS scan + multi-start L-BFGS-B polish (same settings as the
real fit), SSE, and AIC/AICc/BIC differences against double-frozen
(k_single-frozen = 8, k_double-frozen = 11). For the short-lag values the best
fit is also re-solved with tight RK45 and Radau tolerances, as in the Malta
crossover check.

READ, NOT TYPED IN:
  * double-frozen SSE           <- 02_fits/results/double_frozen/vetlyanka_constant_mu_gamma_summary.txt
  * T_perceive used in the paper <- 02_fits/results/single_frozen/vetlyanka_constant_mu_gamma_single_FIXED_fit.csv
Sampling: scipy.stats.qmc.LatinHypercube(seed=LHS_SEED) -- reproducible.
OBSERVATION TIMES: weeks 1..22 at t = 1..22 (same as the fits).

Writes: 06_robustness/results/single_frozen_T_perceive_sensitivity_results.csv
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # .../vetlyanka/06_robustness
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):  # walk up to vetlyanka/, which contains src/
    if os.path.dirname(ROOT) == ROOT:
        raise FileNotFoundError("Could not find the src/ folder above this script.")
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import qmc

from plague_single_sigmoid_model import plague_model
from vetlyanka_bounds import PARAM_NAMES, BOUNDS

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
LEGACY_ALIGNMENT = False
NUM_LHS_SAMPLES = 100000
LHS_SEED = 42
TOP_K = 10
T_PERCEIVE_TEST_VALUES = [0.1, 0.3, 0.5, 1.0, 2.0, 2.5, 5.0, 6.59, 8.0]   # the paper's value is added below
SHORT_LAG_VALUES = [0.1, 0.3, 0.5]                                        # extra solver-robustness check
K_SINGLE_FROZEN, K_DOUBLE_FROZEN = 8, 11

FITS_DIR = os.path.join(ROOT, "02_fits", "results")
DF_SUMMARY = os.path.join(FITS_DIR, "double_frozen", "vetlyanka_constant_mu_gamma_summary.txt")
SF_FIT = os.path.join(FITS_DIR, "single_frozen", "vetlyanka_constant_mu_gamma_single_FIXED_fit.csv")
OUTPUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

t_start, t_end, dt = 0.0, 22.0, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
n_weeks = 22
OBS = slice(0, n_weeks) if LEGACY_ALIGNMENT else slice(1, n_weeks + 1)
initial_conditions = [1699, 1, 0, 0, 0, 0, 0]
X1_C1_PLACEHOLDER = 1.0

_B = dict(zip(PARAM_NAMES, BOUNDS))
GAMMA_CONST_BOUND = (min(_B["gamma1"][0], _B["gamma2"][0]), max(_B["gamma1"][1], _B["gamma2"][1]))
MU_CONST_BOUND = (min(_B["mu1"][0], _B["mu2"][0]), max(_B["mu1"][1], _B["mu2"][1]))
SF_NAMES = ["b1", "b2", "x0", "c", "gamma_const", "mu_const", "sensitivity", "dispose_rate"]
SF_BOUNDS = [_B["b1"], _B["b2"], _B["x0"], _B["c"], GAMMA_CONST_BOUND, MU_CONST_BOUND,
             _B["sensitivity"], _B["dispose_rate"]]

OBSERVED = np.array([3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19, 27, 34,
                     90, 259, 313, 345, 364, 376, 376, 376, 376], dtype=float)


def read_sse(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run the double-frozen fit first.")
    for line in open(path, encoding="utf-8"):
        if line.startswith("SSE="):
            return float(line.split("=", 1)[1])
    raise ValueError(f"No SSE line in {path}")


def read_paper_t_perceive():
    if not os.path.exists(SF_FIT):
        raise FileNotFoundError(f"{SF_FIT} not found -- run the single-frozen fit first.")
    df = pd.read_csv(SF_FIT)
    return float(df.loc[df["Parameter"] == "T_perceive", "Best_Fit"].iloc[0])


def to_full(r, t_perceive):
    d = dict(zip(SF_NAMES, r))
    d["gamma1"] = d["gamma2"] = d["gamma_const"]
    d["mu1"] = d["mu2"] = d["mu_const"]
    d["T_perceive"] = t_perceive
    return np.array([d.get(n, X1_C1_PLACEHOLDER) for n in PARAM_NAMES], dtype=float)


def simulate(r, t_perceive, **solver):
    solver = solver or {"method": "RK45"}
    try:
        sol = solve_ivp(plague_model, [t_start, t_end], initial_conditions,
                        args=(to_full(r, t_perceive),), t_eval=t_points, **solver)
    except (ValueError, FloatingPointError, OverflowError):
        return None
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    w = sol.y[5][::steps_per_week][OBS]
    return w if len(w) == n_weeks else None


def sse(r, t_perceive):
    p = simulate(r, t_perceive)
    if p is None:
        return 1e12
    v = float(np.sum((OBSERVED - p) ** 2))
    return v if np.isfinite(v) else 1e12


def information_criteria(s, k, n=n_weeks):
    aic = n * np.log(s / n) + 2 * k
    return aic, aic + 2 * k * (k + 1) / (n - k - 1), n * np.log(s / n) + k * np.log(n)


def fit_at(t_perceive, samples):
    scores = np.array([sse(r, t_perceive) for r in samples])
    best_s, best_r = np.inf, None
    for i in np.argsort(scores)[:TOP_K]:
        start = samples[i]
        sf = np.maximum(np.abs(start), 1e-3)
        res = minimize(lambda p: sse(p * sf, t_perceive), start / sf, method="L-BFGS-B",
                       bounds=[(lo / s, hi / s) for (lo, hi), s in zip(SF_BOUNDS, sf)],
                       options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 100})
        r = res.x * sf
        v = sse(r, t_perceive)
        if v < best_s:
            best_s, best_r = v, r
    return best_s, best_r


def main():
    df_sse = read_sse(DF_SUMMARY)
    paper_tp = read_paper_t_perceive()
    values = sorted(set(T_PERCEIVE_TEST_VALUES + [round(paper_tp, 6)]))
    aic_df, aicc_df, bic_df = information_criteria(df_sse, K_DOUBLE_FROZEN)

    lo = np.array([b[0] for b in SF_BOUNDS], float); hi = np.array([b[1] for b in SF_BOUNDS], float)
    samples = lo + qmc.LatinHypercube(d=len(SF_BOUNDS), seed=LHS_SEED).random(n=NUM_LHS_SAMPLES) * (hi - lo)

    print(f"Observation times: {'t = 0..21 (LEGACY)' if LEGACY_ALIGNMENT else 't = 1..22 (end of each week)'}")
    print(f"Double-frozen reference: SSE = {df_sse:.6f}  (AIC {aic_df:.3f}, AICc {aicc_df:.3f}, BIC {bic_df:.3f})")
    print(f"T_perceive used in the paper's single-frozen fit: {paper_tp:.6f}")
    print(f"Testing T_perceive = {values}  ({NUM_LHS_SAMPLES} LHS samples, {TOP_K} starts each)\n")

    rows, t0 = [], time.time()
    for tp in values:
        best_s, best_r = fit_at(tp, samples)
        aic_s, aicc_s, bic_s = information_criteria(best_s, K_SINGLE_FROZEN)
        row = {"T_perceive_fixed": tp, "is_paper_value": abs(tp - paper_tp) < 1e-6, "Best_SSE": best_s,
               "dAIC_single_minus_double": aic_s - aic_df, "dAICc_single_minus_double": aicc_s - aicc_df,
               "dBIC_single_minus_double": bic_s - bic_df,
               "single_frozen_loses_on_SSE": best_s > df_sse}
        row.update(dict(zip(SF_NAMES, best_r)))
        print(f"T_perceive = {tp:<9g} SSE = {best_s:10.3f}   dAIC = {aic_s - aic_df:+8.2f}   "
              f"dAICc = {aicc_s - aicc_df:+8.2f}   dBIC = {bic_s - bic_df:+8.2f}"
              f"{'   <- paper value' if row['is_paper_value'] else ''}   [{(time.time() - t0) / 60:.0f} min]",
              flush=True)
        if tp in SHORT_LAG_VALUES:
            check = []
            for label, kw in [("RK45 default", {"method": "RK45"}),
                              ("RK45 tight", {"method": "RK45", "rtol": 1e-10, "atol": 1e-12}),
                              ("Radau tight", {"method": "Radau", "rtol": 1e-10, "atol": 1e-12})]:
                p = simulate(best_r, tp, **kw)
                check.append(np.nan if p is None else float(np.sum((OBSERVED - p) ** 2)))
            spread = (np.nanmax(check) - np.nanmin(check)) / np.nanmin(check)
            row["solver_SSE_spread_pct"] = 100 * spread
            print(f"    solver check: SSE {', '.join(f'{c:.3f}' for c in check)}  (spread {100 * spread:.3f}%)")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "single_frozen_T_perceive_sensitivity_results.csv"), index=False)
    print(f"\n{'=' * 60}\nSUMMARY (double-frozen SSE = {df_sse:.3f})\n{'=' * 60}")
    print(f"Single-frozen SSE range: {df.Best_SSE.min():.2f} to {df.Best_SSE.max():.2f}")
    print(f"Single-frozen loses on SSE at {int(df.single_frozen_loses_on_SSE.sum())}/{len(df)} T_perceive values")
    print(f"AIC favours double-frozen at {int((df.dAIC_single_minus_double > 0).sum())}/{len(df)}, "
          f"BIC at {int((df.dBIC_single_minus_double > 0).sum())}/{len(df)}, "
          f"AICc at {int((df.dAICc_single_minus_double > 0).sum())}/{len(df)}")
    print(f"\nSaved {os.path.join(OUTPUT_DIR, 'single_frozen_T_perceive_sensitivity_results.csv')}")


if __name__ == "__main__":
    main()