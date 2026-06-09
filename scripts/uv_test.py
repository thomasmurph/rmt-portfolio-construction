"""Tests for uv/flake setup, with basic Marchenko-Pastur"""

import io
import numpy as np
import scipy.linalg
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# Numpy check
rng = np.random.default_rng(0)

N, T = 100, 500
q = N / T
lam_minus = (1 - q**0.5) ** 2
lam_plus = (1 + q**0.5) ** 2
X = rng.standard_normal((T, N))

S = X.T @ X / T
# Scipy check
eigs = scipy.linalg.eigvalsh(S)
# Pandas check
print(pd.Series(eigs, name="eigenvalues").describe())
inside = ((eigs > lam_minus) & (eigs < lam_plus)).mean()
# matplotlib check
grid = np.linspace(lam_minus, lam_plus)
density = np.sqrt((lam_plus - grid) * (grid - lam_minus)) / (2 * np.pi * q * grid)
plt.hist(eigs, bins=40, density=True)
plt.plot(grid, density)
plt.savefig("mp_check.png")
# yfinance check
spy = yf.download("SPY", period="5d", auto_adjust=True, progress=False)
print(
    f"yfinance   {yf.__version__}: SPY close {spy['Close'].iloc[-1].item():.2f} ({len(spy)} rows)"
)
