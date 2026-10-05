"""
Reproduce the AR(1) simulation and Figures 2-4 in the paper
Date: September 2026
"""

import argparse
import csv
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from math import erfc, sqrt
from scipy.stats import binomtest

# configuration reported in Section 4.1 of the paper.
N_MONTE_CARLO = 1000
N_BOOTSTRAP = 999
N_SAMPLES = 300
MAX_LAG = 10
BLOCK_LENGTH = int(np.ceil(N_SAMPLES ** (1 / 3)))
ALPHA = 0.05
PHI_X = 0.6
PHI_Y = 0.5
BETA_H1 = 0.25
TRUE_LAG = 2
BURN_IN = 500


def generate_ar1(n, phi, burn_in):
    x = np.zeros(n + burn_in)
    e = np.random.normal(0, 1, size=n + burn_in)
    for t in range(1, n + burn_in):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn_in:]


def generate_coupled(n, phi_x, phi_y, beta, lag, burn_in):
    x = np.zeros(n + burn_in)
    u = np.zeros(n + burn_in)
    ex = np.random.normal(0, 1, size=n + burn_in)
    eu = np.random.normal(0, 1, size=n + burn_in)

    for t in range(1, n + burn_in):
        x[t] = phi_x * x[t - 1] + ex[t]
        u[t] = phi_y * u[t - 1] + eu[t]

    y = np.zeros(n + burn_in)
    for t in range(n + burn_in):
        if t >= lag:
            y[t] = beta * x[t - lag] + u[t]
        else:
            y[t] = u[t]

    return x[burn_in:], y[burn_in:]


def compute_single_ccf(x, y, l):
    if l > 0:
        r = np.corrcoef(x[:-l], y[l:])[0, 1]
    elif l < 0:
        r = np.corrcoef(x[-l:], y[:l])[0, 1]
    else:
        r = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(r):
        raise ValueError(f"Correlation is undefined at lag {l}; check the input series.")
    return r


def compute_max_ccf_grid(x, y, max_lag):
    if len(x) != len(y) or max_lag < 0 or max_lag > len(x) - 2:
        raise ValueError("Series must have equal length and at least two pairs at every lag.")
    lags = range(-max_lag, max_lag + 1)
    corrs = [compute_single_ccf(x, y, l) for l in lags]
    abs_corrs = [abs(r) for r in corrs]
    return np.array(abs_corrs), max(abs_corrs)


def moving_block_bootstrap(ts, block_length):
    n = len(ts)
    if not 1 <= block_length <= n:
        raise ValueError("Block length must be between 1 and the series length.")
    num_blocks = n - block_length + 1
    num_reps = int(np.ceil(n / block_length))
    start_indices = np.random.randint(0, num_blocks, size=num_reps)
    pseudo_ts = []
    for idx in start_indices:
        pseudo_ts.extend(ts[idx:idx + block_length])
    return np.array(pseudo_ts[:n])


def run_trial(args):
    trial_idx, regime, n_bootstrap = args
    np.random.seed(2026 + trial_idx + (9999 if regime == "H1" else 0))

    if regime == "H0":
        x = generate_ar1(N_SAMPLES, PHI_X, BURN_IN)
        y = generate_ar1(N_SAMPLES, PHI_Y, BURN_IN)
    else:
        x, y = generate_coupled(N_SAMPLES, PHI_X, PHI_Y, BETA_H1, TRUE_LAG, BURN_IN)

    grid_corrs, t_sample = compute_max_ccf_grid(x, y, MAX_LAG)

    naive_crit = 1.96 / np.sqrt(N_SAMPLES)
    rej_naive = 1 if t_sample > naive_crit else 0

    r_oracle = abs(compute_single_ccf(x, y, TRUE_LAG))
    rej_oracle = 1 if r_oracle > naive_crit else 0

    p_vals = [
        erfc(r * sqrt(N_SAMPLES / 2.0))
        for r in grid_corrs
    ]
    rej_holm = 1 if min(p_vals) < (ALPHA / len(grid_corrs)) else 0

    boot_stats = []
    for _ in range(n_bootstrap):
        xb = moving_block_bootstrap(x, BLOCK_LENGTH)
        yb = moving_block_bootstrap(y, BLOCK_LENGTH)
        _, tb = compute_max_ccf_grid(xb, yb, MAX_LAG)
        boot_stats.append(tb)

    boot_stats = np.array(boot_stats)
    mbb_p_val = (1.0 + np.sum(boot_stats >= t_sample)) / (n_bootstrap + 1.0)
    rej_mbb = 1 if mbb_p_val < ALPHA else 0

    return {
        "rej_naive": rej_naive,
        "rej_oracle": rej_oracle,
        "rej_holm": rej_holm,
        "rej_mbb": rej_mbb,
        "t_sample": t_sample if trial_idx == 0 else None,
        "boot_stats": boot_stats if trial_idx == 0 else None
    }


def exact_binomial_interval(rejections, trials):
    # return the two-sided Clopper-Pearson 95% interval
    interval = binomtest(rejections, trials).proportion_ci(method="exact")
    return interval.low, interval.high


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=N_MONTE_CARLO)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--workers", type=int, default=min(cpu_count(), 8))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output")
    args = parser.parse_args()
    if args.trials < 1 or args.bootstrap < 10 or args.workers < 1:
        parser.error("trials and workers must be positive; bootstrap must be at least 10")
    return args


if __name__ == "__main__":
    args = parse_args()
    figure_dir = args.output_dir / "figures"
    table_dir = args.output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running Monte Carlo: R={args.trials}, B={args.bootstrap}, Workers={args.workers}")
    t0 = time.time()

    with Pool(processes=args.workers) as pool:
        h0_res = pool.map(run_trial, [(i, "H0", args.bootstrap) for i in range(args.trials)])
        h1_res = pool.map(run_trial, [(i, "H1", args.bootstrap) for i in range(args.trials)])

    methods = ["rej_naive", "rej_oracle", "rej_holm", "rej_mbb"]
    h0_counts = {m: sum(r[m] for r in h0_res) for m in methods}
    h1_counts = {m: sum(r[m] for r in h1_res) for m in methods}
    fpr = {m: h0_counts[m] / args.trials for m in methods}
    tpr = {m: h1_counts[m] / args.trials for m in methods}
    fpr_intervals = {m: exact_binomial_interval(h0_counts[m], args.trials) for m in methods}
    tpr_intervals = {m: exact_binomial_interval(h1_counts[m], args.trials) for m in methods}

    print("\nResults:")
    print(f"{'Method':<15} | {'FPR (H0)':<10} | {'TPR (H1)':<10}")
    print("-" * 40)
    for m in methods:
        print(f"{m:<15} | {fpr[m]:<10.3f} | {tpr[m]:<10.3f}")

    with (table_dir / "results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "FPR", "TPR"])
        for m in methods:
            writer.writerow([m, fpr[m], tpr[m]])

    # Figure 2: rejection proportions and exact binomial intervals.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Naive max", "Fixed-lag IID", "Holm (IID)", "MBB"]
    colors = ["#bb7068", "#668fa3", "#bca275", "#719783"]
    for ax, rates, intervals, title in (
        (axes[0], fpr, fpr_intervals, "Empirical size under $H_0$"),
        (axes[1], tpr, tpr_intervals, "Rejection proportion under $H_1$"),
    ):
        values = np.array([rates[m] for m in methods])
        lows = np.array([intervals[m][0] for m in methods])
        highs = np.array([intervals[m][1] for m in methods])
        errors = np.vstack((values - lows, highs - values))
        ax.bar(labels, values, color=colors, width=0.58,
               yerr=errors, capsize=3, error_kw={"elinewidth": 1})
        ax.set_title(title)
        ax.set_ylabel("Rejection proportion")
        ax.set_ylim(0, 1.12)
        ax.tick_params(axis="x", labelrotation=15)
        for i, value in enumerate(values):
            ax.text(i, highs[i] + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    axes[0].axhline(ALPHA, color="#c44e45", linestyle="--", label=f"Nominal level: {ALPHA}")
    axes[0].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(figure_dir / "fig2_results.png", dpi=300)
    plt.close()

    # Figure 3: bootstrap distribution for the first null trial.
    sample_boot = h0_res[0]["boot_stats"]
    sample_t = h0_res[0]["t_sample"]

    plt.figure(figsize=(7, 4))
    plt.hist(sample_boot, bins=25, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(sample_t, color='crimson', linestyle='--', linewidth=2,
                label=f"Observed statistic: {sample_t:.4f}")
    critical_value = np.percentile(sample_boot, 95)
    plt.axvline(critical_value, color='black', linestyle=':', linewidth=1.5,
                label=f"95th percentile: {critical_value:.4f}")
    plt.xlabel("Bootstrap maximum absolute correlation")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "fig3_distribution.png", dpi=300)
    plt.close()

    # Figure 4: running percentile for one fixed dataset.
    mbb_crit_tracker = [np.percentile(sample_boot[:i], 95) for i in range(10, len(sample_boot) + 1)]
    final_crit = mbb_crit_tracker[-1]

    plt.figure(figsize=(7, 4))
    plt.plot(range(10, len(sample_boot) + 1), mbb_crit_tracker, color='purple', alpha=0.8, linewidth=1.5,
             label='Running 95th percentile')
    plt.axhline(final_crit, color='darkviolet', linestyle=':', label=f'Final Estimate ({final_crit:.4f})')
    plt.xlabel("Number of bootstrap replications")
    plt.ylabel("Estimated 95th percentile")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(figure_dir / "fig4_bootstrap.png", dpi=300)
    plt.close()

    print(f"\nDone in {(time.time() - t0) / 60:.1f} mins. Saved results and figures to {args.output_dir}.")
