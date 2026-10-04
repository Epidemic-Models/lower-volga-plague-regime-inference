"""
vetlyanka_bounds.py

SINGLE SOURCE OF TRUTH for the Vetlyanka standalone double-sigmoid
parameter bounds. Every script that needs these bounds -- LHS sampling,
L-BFGS-B polishing, the null-test library expansion -- should import
PARAMETERS from here rather than keeping its own hardcoded copy.

This file exists specifically because keeping separate hardcoded copies
in parameter_sampling.py and polish_vetlyanka_fit.py silently drifted
out of sync earlier -- one got widened while the other didn't, and nothing
caught it until the mismatch was diagnosed by hand. Importing from one
place instead of copy-pasting the list removes that failure mode entirely.

Historically grounded (documented disease-course timescales) -- do not
widen without new evidence:
    gamma1: 20-day slowest recovery (early phase)
    gamma2: 5-6 day recovery (peak phase)
    mu1, mu2: early- and peak-phase mortality timing (mu2 upper = 12-hour
              fastest documented death)

No independent historical anchor -- free to adjust based on where fits
push against them:
    b1, b2, x0, x1, c, c1, T_perceive, sensitivity, dispose_rate
    (dispose_rate is the one exception in this group worth flagging: it
    DOES correspond to a real physical quantity, body-disposal rate, even
    though it has no single documented historical value -- see its own
    note below, since that made its widening history different from the
    others.)

WIDENING ROUND 1, after the single-sigmoid fit came back pinned on 8 of
its 11 parameters against an earlier, narrower set of bounds (the double-
sigmoid fit was also still pinning b2 and c1 at the time). Every no-anchor
parameter either fit was pinning against got widened together, rather
than only widening whichever side happened to be losing the model
comparison:
    b1:           0.01-0.10   -> 0.01-0.20   (single pinned upper)
    b2:           5.00-13.00  -> 5.00-18.00  (double pinned upper)
    c:            0.2-1.0     -> 0.05-2.0    (double pins lower, single
                                              pins upper -- both ends
                                              widened)
    c1:           0.8-1.5     -> 0.8-2.5     (double pinned upper)
    dispose_rate: 3.51-14     -> 3.51-20     (single pinned upper)

WIDENING ROUND 2, after a subsequent double-sigmoid fit came back pinning
on b1 (new, at the lower wall this time), c1, and dispose_rate again,
even after round 1:
    b1:           0.01-0.20   -> 0.001-0.20  (double pinned LOWER wall)
    c1:           0.8-2.5     -> 0.8-4.0     (double pinned upper)
    dispose_rate: 3.51-20     -> 3.51-30 (temporarily -- see correction below)

dispose_rate CORRECTION (physical constraint, not a fitting-driven
widening): round 2's 30/week ceiling was checked against real time units
(days = 7/rate) and found indefensible -- 30/week implies body disposal
in under 6 hours, not plausible for an 1878 rural village, especially
during the outbreak's worst weeks when gravediggers themselves may be
sick or dead. Reverted to 3.51-7.5, which brackets roughly 1-2 days
(7.5/week ≈ 22.4 hours, close to the original pre-session bound of
7.04/week ≈ 24 hours). If dispose_rate pins at 7.5 again, that should be
treated as a genuine finding (disposal happening as fast as physically
plausible during the crisis), not chased with further widening, the same
way gamma1/gamma2/mu1/mu2 pinning is treated as informative rather than
a bounds problem.

T_perceive went through several revisions, independent of the above (its
own paragraph, since the reasoning was different):
    1.5-3.0  -> 1.5-5.0: single-sigmoid pinned at 3.0, widened for headroom.
    1.5-5.0  -> 0.5-1:   tried tightening based on a hypothesis that fast
                         word-of-mouth in a small village implies a short
                         perception lag. This backfired -- both models'
                         SSE collapsed (single 18x worse, double 1.6x
                         worse) and both re-pinned at the new upper wall,
                         showing the fits wanted MORE time, not less.
    0.5-1    -> 0.5-8:   reverted the tightening, current setting. 8 weeks
                         is generously non-restrictive (over a third of
                         the 22-week observation window) without
                         presupposing a specific perception timescale.
                         This is now settled: a real single-sigmoid fit
                         converged to T_perceive=6.59 weeks, comfortably
                         interior, not pinned at either wall.

Separately: a diagnostic (plot_beta_eff_diagnostic.py, comparing raw
beta(t) to beta_eff(t) = beta(t)*exp(-sensitivity*P(t))) showed that
T_perceive, together with sensitivity, lets the single-sigmoid model
produce a genuine decline in *effective* transmission via behavioral
feedback alone, even though its raw beta(t) is monotonic rise-only. That
means the ordinary single-sigmoid fit isn't a clean test of "no second
regime" -- see PARAMETERS_NULL below for the stricter comparison model
this motivated (feedback disabled entirely, not just re-bounded).
"""

PARAMETERS = [
    ('b1', 0.001, 0.20),
    ('b2', 5.00, 25.00),
    ('x0', 5, 20),
    ('x1', 14, 20),
    ('c', 0.02, 2.0),
    ('c1', 0.8, 4.0),
    ('gamma1', 0.35, 0.7),
    ('gamma2', 1.167, 1.4),
    ('mu1', 1.167, 1.4),
    ('mu2', 1.75, 14),
    ('T_perceive', 0.5, 20.0),
    ('sensitivity', 0.005, 0.2),
    ('dispose_rate', 2.0, 7.5),
]

PARAM_NAMES = [p[0] for p in PARAMETERS]
BOUNDS = [(p[1], p[2]) for p in PARAMETERS]
K_DOUBLE = 13
K_SINGLE = 11  # single-sigmoid drops x1, c1

_DROPPED_FOR_SINGLE = {'x1', 'c1'}
PARAMETERS_SINGLE = [p for p in PARAMETERS if p[0] not in _DROPPED_FOR_SINGLE]
PARAM_NAMES_SINGLE = [p[0] for p in PARAMETERS_SINGLE]
BOUNDS_SINGLE = [(p[1], p[2]) for p in PARAMETERS_SINGLE]
assert len(PARAMETERS_SINGLE) == K_SINGLE

_DROPPED_FOR_NULL = {'x1', 'c1', 'T_perceive', 'sensitivity'}
PARAMETERS_NULL = [p for p in PARAMETERS if p[0] not in _DROPPED_FOR_NULL]
PARAM_NAMES_NULL = [p[0] for p in PARAMETERS_NULL]
BOUNDS_NULL = [(p[1], p[2]) for p in PARAMETERS_NULL]
K_NULL = 9
assert len(PARAMETERS_NULL) == K_NULL