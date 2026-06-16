"""
Estimators for covariance
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


def g_iw(z, q, kappa):
    "Stieltjes transform of a sample matrix whose true C is inverse-Wishart(kappa)"
    lp = ((1 + q) * kappa + 1 + np.sqrt((2 * kappa + 1) * (2 * q * kappa + 1))) / kappa
    lm = ((1 + q) * kappa + 1 - np.sqrt((2 * kappa + 1) * (2 * q * kappa + 1))) / kappa
    return (
        z * (1 + kappa) - kappa * (1 - q) - np.sqrt(z - lp + 0j) * np.sqrt(z - lm + 0j)
    ) / (z * (z + 2 * q * kappa))


def rie_cov(window):
    """
    Rotationally invariant estimator, Algorithm 1 in Bun/Bouchaud/Potters 2017.
    """
    T, N = window.shape
    q = N / T
    vols = window.std().values
    corr = window.corr().values

    lam, vecs = np.linalg.eigh(corr)
    nz = lam > 1e-10  # when T <= N the correlation matrix has exact zero modes

    # inverse-Wishart kappa fitted so the model's lower edge hits our smallest
    # eigenvalue; comes out negative if the spectrum is narrower than pure noise,
    # in which case no correction is needed anyway
    lmin = lam[nz][0]
    kappa = 2 * lmin / ((1 - q - lmin) ** 2 - 4 * q * lmin)
    alpha = 1 / (1 + 2 * q * kappa)

    cleaned = []
    for i in np.where(nz)[0]:
        z = lam[i] - 1j / np.sqrt(N)
        s = (1.0 / (z - np.delete(lam, i))).sum() / (N - 1)
        xi_i = lam[i] / abs(1 - q + q * z * s) ** 2
        if q < 1 and kappa > 0 and lam[i] < 1:
            gamma = (1 + alpha * (lam[i] - 1)) / (
                lam[i] / abs(1 - q + q * z * g_iw(z, q, kappa)) ** 2
            )
            if gamma > 1:
                xi_i *= gamma
        cleaned.append(xi_i)

    xi = np.zeros(N)
    xi[nz] = np.sort(cleaned)
    if (~nz).any():
        # q > 1: the kernel sends zero eigenvalues to zero, which would keep the
        # matrix singular. No principled fix exists (Bun et al sec 9.4); they
        # suggest a common value set by trace = N, like clipping does
        rem = (N - xi[nz].sum()) / (~nz).sum()
        xi[~nz] = rem if rem > 0 else xi[nz].min()
    xi *= N / xi.sum()

    corr_clean = vecs @ np.diag(xi) @ vecs.T
    d = np.sqrt(np.diag(corr_clean))
    corr_clean = corr_clean / np.outer(d, d)

    cov = corr_clean * np.outer(vols, vols)
    return pd.DataFrame(cov, index=window.columns, columns=window.columns)
