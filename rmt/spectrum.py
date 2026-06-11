"""
Simple implementation of Marchenko-Pastur distribution for iid noise eigenvalues
of a correlation matrix.
"""

import numpy as np


def mp_edges(q):
    """Returns (λ-, λ+), the bottom and top edges for ratio N/T = q"""
    return (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2


def mp_density(lam, q):
    """PDF of the Marchenko-Pastur distribution. Support [λ-, λ+]"""
    low, high = mp_edges(q)
    lam = np.clip(lam, low, high)
    rho = np.sqrt((high - lam) * (lam - low)) / (2 * np.pi * q * lam)
    return rho
