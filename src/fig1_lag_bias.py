"""
Generate the illustrative null lag scan in Figure 1
Date: September 2026
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def generate_ar1(n, phi, burn_in=500):
    x = np.zeros(n + burn_in)
    e = np.random.normal(0, 1, size=n + burn_in)
    for t in range(1, n + burn_in):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn_in:]

np.random.seed(2026)
n_samples = 300
x_null = generate_ar1(n_samples, phi=0.6)
y_null = generate_ar1(n_samples, phi=0.5)

lags = range(-10, 11)
corrs = []
for l in lags:
    if l > 0:
        corrs.append(np.corrcoef(x_null[:-l], y_null[l:])[0, 1])
    elif l < 0:
        corrs.append(np.corrcoef(x_null[-l:], y_null[:l])[0, 1])
    else:
        corrs.append(np.corrcoef(x_null, y_null)[0, 1])

naive_threshold = 1.96 / np.sqrt(n_samples)

plt.figure(figsize=(8, 5))
plt.plot(list(lags), corrs, marker='o', color='steelblue', label='Observed Correlation Grid')
plt.axhline(naive_threshold, color='crimson', linestyle='--', label=f'IID single-lag threshold ({naive_threshold:.3f})')
plt.axhline(-naive_threshold, color='crimson', linestyle='--')
plt.title("Cross-Correlation Lag Sweep Under Independence", fontsize=10)
plt.xlabel("Temporal Lag Window")
plt.ylabel("Cross-Correlation Coefficient")
plt.legend(fontsize=8)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
figure_dir = Path(__file__).resolve().parents[1] / "output" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(figure_dir / "fig1_lag_bias.png", dpi=300)
