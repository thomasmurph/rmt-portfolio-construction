"""Plot the eigenvalue spectrum from 2001-2004 alongside MP"""

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rmt.spectrum import mp_density, mp_edges

returns = pd.read_csv("data/returns.csv.gz", index_col=0, parse_dates=True)
window = returns.loc["2001-01-04":"2004-12-31"]
T, N = window.shape
q = N / T

eigenvalues = np.linalg.eigvalsh(window.corr().values)

# variance with market mode removed, as in Laloux 1999
sigma2 = 1 - eigenvalues[-1] / N

high = mp_edges(q)[1]

# least squares grid search to fit the best sigma2 and q to the histogram of the eigenvalues
counts, bin_edges = np.histogram(
    eigenvalues, bins=60, range=(0, 2 * high), density=True
)
centers = (bin_edges[:-1] + bin_edges[1:]) / 2
best_loss = np.inf
s2_fit, q_fit = sigma2, q
for s2 in np.linspace(0.3, 1.0, 71):
    for qe in np.linspace(0.1, 0.9, 81):
        loss = ((mp_density(centers / s2, qe) / s2 - counts) ** 2).sum()
        if loss < best_loss:
            s2_fit, q_fit, best_loss = s2, qe, loss

print(
    f"q = {q:.3f}, largest eigenvalue = {eigenvalues[-1]:.1f}, "
    f"{(eigenvalues > high).sum()} above the bulk edge {high:.2f}"
)
print(f"market mode removed: sigma2 = {sigma2:.2f}")
print(f"fit: sigma2 = {s2_fit:.2f}, q_eff = {q_fit:.2f}")

x = np.linspace(0, 2 * high, 500)
plt.hist(eigenvalues, bins=100, range=(0, 2 * high), density=True, label="eigenvalues")
plt.plot(x, mp_density(x, q), label=f"Marchenko-Pastur, q={q:.2f}")
plt.plot(
    x,
    mp_density(x / sigma2, q) / sigma2,
    label=f"MP, market removed (sigma2={sigma2:.2f})",
)
plt.plot(
    x,
    mp_density(x / s2_fit, q_fit) / s2_fit,
    label=f"MP fit (sigma2={s2_fit:.2f}, q={q_fit:.2f})",
)
plt.xlabel("eigenvalue")
plt.ylabel("density")
plt.legend()
Path("figs").mkdir(exist_ok=True)
plt.savefig("figs/spectrum.png")
print("wrote figs/spectrum.png")
