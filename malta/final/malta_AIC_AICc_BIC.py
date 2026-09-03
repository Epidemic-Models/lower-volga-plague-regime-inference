"""
malta_AIC_AICc_BIC.py

Computes AIC, AICc, and BIC for all four confirmed, final Malta models,
mirroring Vetlyanka's own AIC_AICc_BIC_gamma_mu_freeze.py exactly.

k accounting (dispose_rate is FIXED, not searched, for both double
models, following the escape-route resolution -- see SI methodology):
    Single-sigmoid:              k=11 (dispose_rate free)
    Double-sigmoid:               k=12 (dispose_rate fixed at 3.0)
    Single, constant gamma,mu:    k=9  (dispose_rate free)
    Double, constant gamma,mu:    k=10 (dispose_rate fixed at 2.0)

Path pattern matches the executed final/diagnostics/ reorganization:
this script lives in malta/final/, data/ lives at malta/data/ (one
level up from this script, not two).
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "comparisons")
os.makedirs(OUTPUT_DIR, exist_ok=True)

n = 27  # Malta's weekly observations

# Confirmed, final SSE and k values -- update here if any number changes.
MODELS = {
    "Single-sigmoid":            {"SSE": 18961.0,   "k": 11},
    "Double-sigmoid":             {"SSE": 14215.55,  "k": 12},
    "Single, constant gamma,mu":  {"SSE": 13549.0,   "k": 9},
    "Double, constant gamma,mu":  {"SSE": 8953.0,    "k": 10},
}


def calculate_information_criteria(sse, k, n):
    if sse <= 0:
        sse = np.finfo(float).tiny
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    aicc = np.inf if n <= k + 1 else aic + (2 * k * (k + 1)) / (n - k - 1)
    return aic, aicc, bic


def main():
    rows = []
    for name, vals in MODELS.items():
        aic, aicc, bic = calculate_information_criteria(vals["SSE"], vals["k"], n)
        rows.append({"Model": name, "SSE": vals["SSE"], "k": vals["k"],
                     "AIC": aic, "AICc": aicc, "BIC": bic})

    df = pd.DataFrame(rows).set_index("Model")
    print(df.round(2))

    print("\nPairwise deltas (positive favors double, matching Vetlyanka's convention):")
    for pair_name, (single_key, double_key) in [
        ("Sigmoidal pair", ("Single-sigmoid", "Double-sigmoid")),
        ("Constant gamma,mu pair", ("Single, constant gamma,mu", "Double, constant gamma,mu")),
    ]:
        s, d = df.loc[single_key], df.loc[double_key]
        print(f"\n  {pair_name}:")
        print(f"    dAIC  = {s['AIC'] - d['AIC']:.2f}")
        print(f"    dAICc = {s['AICc'] - d['AICc']:.2f}")
        print(f"    dBIC  = {s['BIC'] - d['BIC']:.2f}")

    df.to_csv(os.path.join(OUTPUT_DIR, "malta_model_comparison.csv"))
    print(f"\nSaved {OUTPUT_DIR}/malta_model_comparison.csv")


if __name__ == "__main__":
    main()