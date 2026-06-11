"""
Three estimators for covariance
Input window Dataframe; Output adjusted covariance matrix
"""

import numpy as np
import pandas as pd

from rmt.spectrum import mp_edges


def sample_cov(window):
    "Naive covariance"
    return window.cov()


def clipped_cov(window, sigma2=None):
    "Covariance clipping as described in Laloux 1999"
    T, N = window.shape
    q = N / T
    vols = window.std().values
    corr = window.corr().values

    lam, vecs = np.linalg.eigh(corr)
    if sigma2 is None:
        sigma2 = 1 - lam[-1] / N
    edge = sigma2 * mp_edges(q)[1]

    noise = lam <= edge
    lam_clean = lam.copy()
    lam_clean[noise] = lam[noise].mean()
    corr_clean = vecs @ np.diag(lam_clean) @ vecs.T

    # the rebuild leaves the diagonal slightly off 1, push it back
    d = np.sqrt(np.diag(corr_clean))
    corr_clean = corr_clean / np.outer(d, d)

    cov = corr_clean * np.outer(vols, vols)
    return pd.DataFrame(cov, index=window.columns, columns=window.columns)


def lw_cov(window):
    "Ledoit-Wolf 2004 shrinkage of the sample covariance towards the scaled identity"
    X = window.values - window.values.mean(axis=0)
    T, N = X.shape
    S = X.T @ X / T
    m = np.trace(S) / N
    d2 = ((S - m * np.eye(N)) ** 2).sum() / N
    # average distance between one day's outer product and S
    b2 = ((X**2).sum(axis=1) ** 2).sum() / T**2 / N - (S**2).sum() / T / N
    b2 = min(b2, d2)
    shrink = b2 / d2
    S_star = shrink * m * np.eye(N) + (1 - shrink) * S
    return pd.DataFrame(S_star, index=window.columns, columns=window.columns)
