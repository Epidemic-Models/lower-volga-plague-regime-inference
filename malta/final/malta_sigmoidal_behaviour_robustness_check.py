"""
malta_sigmoidal_behaviour_robustness_check.py

Follow-up diagnostic for the Malta sigmoidal double-sigmoid model.

Purpose
-------
The earlier dispose-rate diagnostic asked only whether fixing dispose_rate
prevents b2 from pinning at its ceiling. This script asks the next
scientific question: once that escape route is controlled, is the
inferred behavioural story robust to the chosen fixed dispose_rate?

For each fixed dispose_rate, the script refits the model, reconstructs
beta_base(t)/beta_eff(t)/reduction(t), and saves both numeric summaries
and overlay plots so the behavioral story can be compared directly
across dispose_rate=3, 4, 5.

Required:
    plague_double_sigmoid_model.py in the vetlyanka/ folder.
"""

import os
import sys
import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import qmc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostics", "behaviour_robustness")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VETLYANKA_DIR = "/Users/bornalid/PycharmProjects/lower-volga-plague-regime-inference/vetlyanka "
sys.path.insert(0, VETLYANKA_DIR)
from plague_double_sigmoid_model import plague_model, double_sigmoid

PARAM_NAMES = ["b1", "b2", "x0", "x1", "c", "c1", "gamma1", "gamma2", "mu1", "mu2",
               "T_perceive", "sensitivity"]
B2_CEILING = 100.0
BOUNDS = [
    (0.001, 8.0), (1.0, B2_CEILING), (1.0, 20.0), (5.0, 27.0), (0.01, 5.0), (0.01, 5.0),
    (0.35, 0.7), (1.0, 1.4), (1.0, 1.4), (1.0, 10.0), (0.1, 20.0), (0.0001, 0.5),
]
K = len(BOUNDS)
IDX_X0, IDX_X1 = 2, 3
X0_X1_MIN_GAP = 1.0

DISPOSE_RATE_VALUES_TO_TEST = [3.0, 4.0, 5.0]

observed = np.array([0, 0, 0, 41, 100, 227, 440, 592, 813, 1139, 1523, 1896,
                      2225, 2609, 2860, 3045, 3269, 3486, 3703, 3902, 4045,
                      4147, 4247, 4309, 4339, 4379, 4400], dtype=float)
n_weeks = len(observed)
t_start, t_end, dt = 0.0, n_weeks - 1, 0.01
t_points = np.arange(t_start, t_end + dt, dt)
steps_per_week = int(round(1.0 / dt))
initial_conditions = [11000, 1, 0, 0, 0, 0, 0]

NUM_LHS_SAMPLES = 40000
TOP_K = 5
LHS_SEED = 42


class TimeoutError_(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError_()


def reduced_to_full(reduced, dispose_rate_fixed):
    b1, b2, x0, x1, c, c1, gamma1, gamma2, mu1, mu2, T_perceive, sensitivity = reduced
    return np.array([b1, b2, x0, x1, c, c1, gamma1, gamma2, mu1, mu2,
                      T_perceive, sensitivity, dispose_rate_fixed], dtype=float)


def simulate_full(reduced, dispose_rate_fixed):
    """Returns the full solve_ivp solution object on t_points."""
    full = reduced_to_full(reduced, dispose_rate_fixed)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(5)
    try:
        sol = solve_ivp(plague_model, [t_start, t_end], initial_conditions,
                         args=(full,), t_eval=t_points, method="RK45",
                         rtol=1e-6, atol=1e-8)
    except (ValueError, FloatingPointError, OverflowError, TimeoutError_):
        return None
    finally:
        signal.alarm(0)
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return None
    return sol


def simulate_weekly(reduced, dispose_rate_fixed):
    """Weekly-downsampled DDI array for SSE fitting -- was missing/undefined
    in the original script (called but never defined, a real bug fixed here)."""
    sol = simulate_full(reduced, dispose_rate_fixed)
    if sol is None:
        return None
    predicted = sol.y[4][::steps_per_week][:n_weeks]
    if predicted.shape[0] != n_weeks or not np.all(np.isfinite(predicted)):
        return None
    return predicted


def sse_unscaled(reduced, dispose_rate_fixed):
    if reduced[IDX_X0] >= reduced[IDX_X1] - X0_X1_MIN_GAP:
        return 1e12
    predicted = simulate_weekly(reduced, dispose_rate_fixed)
    if predicted is None:
        return 1e12
    sse = np.sum((observed - predicted) ** 2)
    return float(sse) if np.isfinite(sse) else 1e12


def scale_parameters(params):
    sf = np.maximum(np.abs(params), 1e-3)
    return params / sf, sf


def fit_at_dispose_rate(dispose_rate_fixed):
    sampler = qmc.LatinHypercube(d=K, seed=LHS_SEED)
    unit = sampler.random(n=NUM_LHS_SAMPLES)
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    samples = lo + unit * (hi - lo)

    scored = [(sse_unscaled(row, dispose_rate_fixed), row) for row in samples]
    scored.sort(key=lambda pair: pair[0])
    top = scored[:TOP_K]

    best_sse, best_params = np.inf, None
    for lhs_sse, start in top:
        sx0, sf = scale_parameters(start)
        scaled_bounds = [(lo_b / f, hi_b / f) for (lo_b, hi_b), f in zip(BOUNDS, sf)]
        result = minimize(lambda sp: sse_unscaled(sp * sf, dispose_rate_fixed), sx0,
                           method="L-BFGS-B", bounds=scaled_bounds,
                           options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10})
        opt = result.x * sf
        polished_sse = sse_unscaled(opt, dispose_rate_fixed)
        if polished_sse < best_sse:
            best_sse, best_params = polished_sse, opt.copy()
    return best_sse, best_params


def reconstruct_behaviour(reduced, dispose_rate_fixed):
    sol = simulate_full(reduced, dispose_rate_fixed)
    if sol is None:
        raise RuntimeError(f"Final simulation failed for dispose_rate={dispose_rate_fixed}")
    b1, b2, x0, x1, c, c1, gamma1, gamma2, mu1, mu2, T_perceive, sensitivity = reduced
    P = sol.y[6]
    beta_base = double_sigmoid(sol.t, b1, b2, x0, x1, c, c1)
    suppression_multiplier = np.exp(-sensitivity * P)
    reduction = 1.0 - suppression_multiplier
    beta_eff = beta_base * suppression_multiplier

    trajectory = pd.DataFrame({
        "time_week": sol.t, "dispose_rate_fixed": dispose_rate_fixed, "P": P,
        "beta_base": beta_base, "suppression_multiplier": suppression_multiplier,
        "reduction_fraction": reduction, "reduction_percent": 100.0 * reduction,
        "beta_eff": beta_eff, "DDI_cumulative_deaths": sol.y[4],
    })

    peak_reduction_idx = int(np.argmax(reduction))
    peak_beta_base_idx = int(np.argmax(beta_base))
    peak_beta_eff_idx = int(np.argmax(beta_eff))
    summary = {
        "peak_beta_base": float(beta_base[peak_beta_base_idx]),
        "week_peak_beta_base": float(sol.t[peak_beta_base_idx]),
        "peak_beta_eff": float(beta_eff[peak_beta_eff_idx]),
        "week_peak_beta_eff": float(sol.t[peak_beta_eff_idx]),
        "peak_reduction_percent": float(100.0 * reduction[peak_reduction_idx]),
        "week_peak_reduction": float(sol.t[peak_reduction_idx]),
        "max_P": float(np.max(P)),
    }
    return trajectory, summary


def save_overlay_plot(all_trajectories):
    fig, ax = plt.subplots(figsize=(10, 6))
    for dr, df in all_trajectories.items():
        ax.plot(df["time_week"], df["beta_base"], linestyle="--", alpha=0.75,
                 label=f"beta_base, dispose_rate={dr:g}")
        ax.plot(df["time_week"], df["beta_eff"], linewidth=2, label=f"beta_eff, dispose_rate={dr:g}")
    ax.set_xlabel("Week"); ax.set_ylabel("Transmission rate (per week)")
    ax.set_title("Malta behavioural robustness: structural vs effective transmission")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "malta_behaviour_robustness_beta_overlay.png"), dpi=300)
    plt.close(fig)


def save_reduction_plot(all_trajectories):
    fig, ax = plt.subplots(figsize=(10, 6))
    for dr, df in all_trajectories.items():
        ax.plot(df["time_week"], df["reduction_percent"], linewidth=2, label=f"dispose_rate={dr:g}")
    ax.set_xlabel("Week"); ax.set_ylabel("Behavioural reduction (%)")
    ax.set_title("Malta behavioural robustness: inferred suppression")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "malta_behaviour_robustness_reduction.png"), dpi=300)
    plt.close(fig)


def main():
    print("Malta sigmoidal double-sigmoid behavioural robustness check")
    print(f"Output directory: {OUTPUT_DIR}\n")
    summary_rows, all_trajectories = [], {}
    header = f"{'dispose_rate':>12} {'SSE':>14} {'b2':>10} {'sens':>12} {'peak red %':>12}"
    print(header); print("-" * len(header))

    for dr in DISPOSE_RATE_VALUES_TO_TEST:
        best_sse, best_params = fit_at_dispose_rate(dr)
        if best_params is None:
            print(f"{dr:>12.2f} {'FIT FAILED':>14}")
            continue
        trajectory, behaviour_summary = reconstruct_behaviour(best_params, dr)
        all_trajectories[dr] = trajectory
        trajectory.to_csv(os.path.join(OUTPUT_DIR, f"trajectory_dispose_rate_{dr:g}.csv"), index=False)

        row = {"dispose_rate_fixed": dr, "SSE": best_sse}
        row.update(zip(PARAM_NAMES, best_params))
        row.update(behaviour_summary)
        summary_rows.append(row)
        print(f"{dr:>12.2f} {best_sse:>14.3f} {best_params[1]:>10.3f} "
              f"{best_params[11]:>12.6f} {behaviour_summary['peak_reduction_percent']:>12.2f}")

    if not summary_rows:
        raise RuntimeError("No successful fits were obtained.")
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, "malta_behaviour_robustness_summary.csv"), index=False)
    save_overlay_plot(all_trajectories)
    save_reduction_plot(all_trajectories)
    print(f"\nSaved summary and plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()