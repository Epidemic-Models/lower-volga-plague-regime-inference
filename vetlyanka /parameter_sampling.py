"""
parameter_sampling.py

Generates Latin Hypercube samples over the full 13-parameter double-sigmoid
space (the superset -- single-sigmoid and frozen-gamma/mu fits reuse this
same file, either ignoring x1/c1 or re-expanding gamma/mu, rather than each
drawing their own separate sample set).

Output goes to data/samples/parameter_samples_<num_samples>.csv, inside
this project's own data/ folder -- not the working directory directly --
so every generated file lands in one predictable place instead of
scattering across vetlyanka/ as each script is run.

This project uses two sample sizes at different points (200,000 for the
original sigmoidal-gamma/mu fits, 400,000 for the frozen-gamma/mu fits,
which search a larger combined parameter space -- see vetlyanka_bounds.py
for why). Suffixing the filename with the count lets both sets coexist
without one silently overwriting the other.

Every script that consumes this file should build the same path
(f"data/samples/parameter_samples_{NUM_SAMPLES}.csv") rather than a bare
filename, so it always finds the file regardless of which folder it's
run from within the project.
"""

import os
import numpy as np
from pydoe import lhs  # Latin Hypercube Sampling
from vetlyanka_bounds import PARAMETERS  # single shared source of truth --
                                          # see vetlyanka_bounds.py

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples")


def generate_scaled_lhs_samples(num_params: int, num_samples: int, parameters: list,
                                 seed: int = 42) -> str:
    """
    Generate scaled Latin Hypercube samples and save them to
    data/samples/parameter_samples_<num_samples>.csv.

    Parameters:
    - num_params (int): Number of parameters
    - num_samples (int): Number of samples to generate -- also becomes part
      of the output filename, see module docstring
    - parameters (list): List of (name, min, max) tuples
    - seed (int): Random seed. pyDOE's lhs() draws from numpy's global RNG,
      so fixing the seed here is what makes the sample set (and therefore
      the LHS-best starting point, and therefore the polished fit)
      reproducible run to run and machine to machine. Report this seed
      value in the paper / methods so results are exactly reproducible by
      reviewers.

    Returns the output path actually written, so calling code can use
    it directly rather than reconstructing it.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    output_file = os.path.join(DATA_DIR, f"parameter_samples_{num_samples}.csv")

    np.random.seed(seed)

    # Generate unscaled samples
    lhs_samples = lhs(num_params, samples=num_samples)

    # Scale them according to parameter bounds
    scaled_samples = [
        [row[j] * (parameters[j][2] - parameters[j][1]) + parameters[j][1] for j in range(num_params)]
        for row in lhs_samples
    ]

    scaled_array = np.array(scaled_samples)

    # Save with parameter names as header
    header = ",".join([p[0] for p in parameters])
    np.savetxt(output_file, scaled_array, delimiter=",", header=header, comments="")
    print(f"Scaled parameter samples saved to {output_file}")
    print(f"(generated with random seed={seed} -- rerunning this script with the same seed "
          f"and the same num_samples reproduces this exact sample set; a DIFFERENT "
          f"num_samples produces a genuinely different, non-nested sample set, not a "
          f"superset/subset of this one, since LHS stratifies the space based on the "
          f"requested sample count itself)")

    return output_file


# Example usage
if __name__ == "__main__":
    NUM_SAMPLES = 400000  # set to 400000 to generate the frozen-fit sample set

    generate_scaled_lhs_samples(
        num_params=len(PARAMETERS),
        num_samples=NUM_SAMPLES,
        parameters=PARAMETERS,
    )