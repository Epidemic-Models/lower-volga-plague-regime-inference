"""
parameter_sampling.py

Generates Latin Hypercube samples over the full 13-parameter double-sigmoid
space. These files are read by the double, single and double-frozen fits
(the single fit ignores x1/c1; double-frozen averages gamma1/gamma2 and
mu1/mu2 into gamma_const/mu_const). The null and single-frozen fits draw
their own LHS inside their own scripts and do NOT use these files.

Output goes to this script's own folder (01_sampling/):
    parameter_samples_200000.csv  -- double, single, sigmoidal bootstrap library
    parameter_samples_400000.csv  -- double-frozen fit

REPRODUCIBILITY: the seed is passed directly to lhs(). pyDOE >= 1.0
IGNORES np.random.seed(), so the earlier version of this script produced a
different sample set on every run. With the fix below, the same seed and
the same pyDOE version always give identical files. Report the seed (42)
and the pyDOE version in the paper / README.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # .../vetlyanka/01_sampling
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):  # walk up to vetlyanka/, which contains src/
    if os.path.dirname(ROOT) == ROOT:
        raise FileNotFoundError("Could not find the src/ folder above this script.")
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
from pydoe import lhs  # Latin Hypercube Sampling
from vetlyanka_bounds import PARAMETERS  # single shared source of truth (src/vetlyanka_bounds.py)

DATA_DIR = HERE  # samples are saved next to this script, in 01_sampling/


def generate_scaled_lhs_samples(num_params: int, num_samples: int, parameters: list,
                                seed: int = 42) -> str:
    """
    Generate scaled Latin Hypercube samples and save them to
    01_sampling/parameter_samples_<num_samples>.csv.

    Parameters:
    - num_params (int): Number of parameters
    - num_samples (int): Number of samples to generate (also part of the filename)
    - parameters (list): List of (name, min, max) tuples
    - seed (int): Random seed, passed directly to lhs() so the sample set is
      identical on every run and every machine (given the same pyDOE version).

    Returns the output path actually written.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    output_file = os.path.join(DATA_DIR, f"parameter_samples_{num_samples}.csv")

    # Generate unscaled samples, reproducibly.
    # pyDOE >= 1.0 ignores np.random.seed, so pass the seed to lhs() directly;
    # old pyDOE (0.3.x) has no random_state argument, so fall back to the global seed.
    try:
        lhs_samples = lhs(num_params, samples=num_samples, random_state=seed)
    except TypeError:
        np.random.seed(seed)
        lhs_samples = lhs(num_params, samples=num_samples)

    # Scale them according to parameter bounds
    lo = np.array([p[1] for p in parameters])
    hi = np.array([p[2] for p in parameters])
    scaled_array = lo + lhs_samples * (hi - lo)

    # Save with parameter names as header
    header = ",".join([p[0] for p in parameters])
    np.savetxt(output_file, scaled_array, delimiter=",", header=header, comments="")
    print(f"Scaled parameter samples saved to {output_file} (seed={seed})")

    return output_file


if __name__ == "__main__":
    # 200,000: double and single fits + sigmoidal bootstrap library
    # 400,000: double-frozen fit
    for NUM_SAMPLES in (200000, 400000):
        generate_scaled_lhs_samples(
            num_params=len(PARAMETERS),
            num_samples=NUM_SAMPLES,
            parameters=PARAMETERS,
        )