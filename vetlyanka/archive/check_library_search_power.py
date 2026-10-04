"""
check_library_search_power.py

Diagnostic: how good is the trajectory-library "refit" (used inside
parametric_bootstrap_test.py / bootstrap_frozen.py -- pick whichever
precomputed candidate best matches a dataset) compared to the REAL
fitting procedure (multi-start LHS + L-BFGS-B polishing) that actually
produced your reported SSE values?

If the library-search SSE against the real, observed data is
substantially worse than the real fit's SSE, that means the bootstrap's
"refit" step is a systematically weaker stand-in for your actual
fitting procedure -- and if that gap differs between double and single,
it could bias the bootstrap's null distribution in a way that has
nothing to do with whether a genuine second regime exists.

This runs the identical library-search logic the bootstrap uses, but
against the REAL observed data instead of a synthetic replicate, so the
result is directly comparable to your real, polished fit SSE.

DATA LAYOUT: reads both sigmoidal trajectory libraries from
data/bootstrap/sigmoidal/.
"""

import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP_DIR = os.path.join(BASE_DIR, "data", "bootstrap", "sigmoidal")

# Your real, multi-start-polished fit results -- for direct comparison.
REAL_DOUBLE_SSE = 393.136
REAL_SINGLE_SSE = 960.585

observed = np.array([3, 3, 5, 5, 6, 6, 8, 11, 11, 12, 19, 27, 34,
                      90, 259, 313, 345, 364, 376, 376, 376, 376], dtype=float)

print(f"Loading trajectory libraries from {BOOTSTRAP_DIR} ...")
double_trajectories = np.load(os.path.join(BOOTSTRAP_DIR, "double_trajectories.npy"))
single_trajectories = np.load(os.path.join(BOOTSTRAP_DIR, "single_trajectories.npy"))

double_valid = ~np.isnan(double_trajectories).any(axis=1)
single_valid = ~np.isnan(single_trajectories).any(axis=1)
double_trajectories = double_trajectories[double_valid]
single_trajectories = single_trajectories[single_valid]
print(f"Using {double_trajectories.shape[0]} valid double-sigmoid candidates, "
      f"{single_trajectories.shape[0]} valid single-sigmoid candidates.\n")

sse_double = np.sum((double_trajectories - observed) ** 2, axis=1)
sse_single = np.sum((single_trajectories - observed) ** 2, axis=1)

library_best_double = float(sse_double.min())
library_best_single = float(sse_single.min())

print("=" * 60)
print("Library-search 'best match' vs. real, polished fit")
print("=" * 60)
print(f"{'Model':<10} {'Library-search SSE':>20} {'Real polished SSE':>20} {'Gap':>10} {'Gap %':>8}")
for name, lib_sse, real_sse in [("Double", library_best_double, REAL_DOUBLE_SSE),
                                  ("Single", library_best_single, REAL_SINGLE_SSE)]:
    gap = lib_sse - real_sse
    gap_pct = 100 * gap / real_sse
    print(f"{name:<10} {lib_sse:>20.3f} {real_sse:>20.3f} {gap:>10.3f} {gap_pct:>7.1f}%")

print("\nInterpretation:")
print("- A large positive gap means the library search is a meaningfully")
print("  weaker stand-in for the real fitting procedure for that model.")
print("- If the GAP DIFFERS substantially between double and single, that")
print("  asymmetry could bias the bootstrap's null distribution -- worth")
print("  checking directly against how large a shift would be needed to")
print("  explain the observed p-value pattern.")