"""
Generate Appendix B's contemporaneous-coupling illustration
Date: October 2026
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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

np.random.seed(2026)
n_samples = 300
burn_in = 500

x_full = np.zeros(n_samples + burn_in)
u_full = np.zeros(n_samples + burn_in)
e_x = np.random.normal(0, 1, n_samples + burn_in)
e_u = np.random.normal(0, 1, n_samples + burn_in)

for t in range(1, n_samples + burn_in):
    x_full[t] = 0.6 * x_full[t-1] + e_x[t]
    u_full[t] = 0.5 * u_full[t-1] + e_u[t]

x = x_full[burn_in:]
u = u_full[burn_in:]
# construct contemporaneous coupling; -0.3 is a coefficient, not a correlation.
y = -0.3 * x + u

# compute the lag scan.
lags = range(-10, 11)
sample_corrs = []
for l in lags:
    if l > 0: sample_corrs.append(np.corrcoef(x[:-l], y[l:])[0, 1])
    elif l < 0: sample_corrs.append(np.corrcoef(x[-l:], y[:l])[0, 1])
    else: sample_corrs.append(np.corrcoef(x, y)[0, 1])

# calibrate the maximum absolute statistic under process independence.
b_reps = 999
boot_max = []
for _ in range(b_reps):
    xb = moving_block_bootstrap(x, 7)
    yb = moving_block_bootstrap(y, 7)
    bc = []
    for l in lags:
        if l > 0: bc.append(np.corrcoef(xb[:-l], yb[l:])[0, 1])
        elif l < 0: bc.append(np.corrcoef(xb[-l:], yb[:l])[0, 1])
        else: bc.append(np.corrcoef(xb, yb)[0, 1])
    boot_max.append(max(np.abs(bc)))

crit_val = np.percentile(boot_max, 95)

plt.figure(figsize=(9, 5))
plt.plot(list(lags), sample_corrs, marker='o', color='steelblue', label="Observed Cross-Correlation Scan")
plt.axhline(crit_val, color='crimson', linestyle='--', label=f'MBB 95% Max-Statistic Threshold (±{crit_val:.3f})')
plt.axhline(-crit_val, color='crimson', linestyle='--')
plt.xlabel("Temporal Lag Window")
plt.ylabel("Cross-Correlation Coefficient")
plt.title("Appendix Diagnostic under Contemporaneous Coupling")
plt.legend(fontsize=8)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
figure_dir = Path(__file__).resolve().parents[1] / "output" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(figure_dir / "fig5.png", dpi=300)
