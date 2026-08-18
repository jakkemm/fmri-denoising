from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from sklearn.decomposition._pca import _assess_dimension

from utils.constants import FloatArray


def _mp_bounds(gamma):
    """Return the bounds of the Marchenko-Pastur distribution."""
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must satisfy 0 < gamma <= 1")

    sqrt_gamma = np.sqrt(gamma)

    b_minus = (1.0 - sqrt_gamma) ** 2
    b_plus = (1.0 + sqrt_gamma) ** 2

    return b_minus, b_plus


def _mp_pdf(nu, gamma):
    """Marchenko-Pastur probability density."""
    b_minus, b_plus = _mp_bounds(gamma)

    if nu <= b_minus or nu >= b_plus:
        return 0.0

    numerator = np.sqrt((b_plus - nu) * (nu - b_minus))
    denominator = 2.0 * np.pi * gamma * nu

    return numerator / denominator


def _mp_cdf(nu, gamma):
    """Numerically evaluate the Marchenko-Pastur CDF."""
    b_minus, b_plus = _mp_bounds(gamma)

    if nu <= b_minus:
        return 0.0

    if nu >= b_plus:
        return 1.0

    value, _ = quad(
        lambda x: _mp_pdf(x, gamma),
        b_minus,
        nu,
    )

    return value


def _mp_ppf(probability, gamma):
    """Numerically evaluate the inverse Marchenko-Pastur CDF."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must satisfy 0 < probability < 1")

    b_minus, b_plus = _mp_bounds(gamma)

    return brentq(
        lambda nu: _mp_cdf(nu, gamma) - probability,
        b_minus,
        b_plus,
    )


def _mp_quantiles(n_components, gamma):
    """
    Return expected MP eigenvalue quantiles in descending order.

    The returned ordering matches an eigenspectrum sorted from
    largest to smallest eigenvalue.
    """
    probabilities = (np.arange(1, n_components + 1) - 0.5) / n_components

    quantiles = np.array([_mp_ppf(probability, gamma) for probability in probabilities])
    return quantiles[::-1]


@dataclass
class ModelOrderSelectionResult:
    q_hat: int

    eigenvalues: FloatArray
    eigenvectors: FloatArray

    expected_noise_spectrum: FloatArray
    adjusted_eigenvalues: FloatArray

    log_evidence: FloatArray
    

@dataclass
class ModelOrderSelection:
    def find_best_q(self, X: FloatArray):
        """Estimate the PICA model order."""
        n_timepoints, n_samples = X.shape

        gamma = n_timepoints / n_samples

        covariance = (X @ X.T) / n_samples
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # np.linalg.eigh returns eigenvalues in ascending order
        order = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Expected finite-sample eigenspectrum for isotropic Gaussian noise
        expected_noise_spectrum = _mp_quantiles(
            n_components=n_timepoints,
            gamma=gamma,
        )

        # finite-sample eigenspectrum adjustment
        adjusted_eigenvalues = eigenvalues / expected_noise_spectrum

        candidate_q = np.arange(1, n_timepoints)

        log_evidence = np.array([
            _assess_dimension(spectrum=adjusted_eigenvalues, rank=int(q), n_samples=n_samples)
            for q in candidate_q
        ])

        # candidate_q starts at 1, whereas argmax starts at 0.
        best_idx = int(np.argmax(log_evidence))
        q_hat = int(candidate_q[best_idx])
        
        return ModelOrderSelectionResult(
            q_hat=q_hat,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            expected_noise_spectrum=expected_noise_spectrum,
            adjusted_eigenvalues=adjusted_eigenvalues,
            log_evidence=log_evidence,
        )