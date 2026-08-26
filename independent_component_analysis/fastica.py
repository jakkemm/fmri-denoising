from dataclasses import dataclass

import numpy as np

from utils.constants import FloatArray


@dataclass
class FastICAResult:
    unmixing: FloatArray
    sources: FloatArray

    n_iter: int
    converged: bool
    convergence_value: float


class SymmetricFastICA:
    def __init__(self, tolerance=1e-5, max_iter=1000, eigenvalue_floor=1e-12):
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.eigenvalue_floor = eigenvalue_floor

    def fit_transform(self, Z: FloatArray) -> FastICAResult:
        n_components, n_samples = Z.shape

        Q = np.eye(n_components, dtype=float)

        convergence_value = np.inf
        converged = False

        for iteration in range(1, self.max_iter + 1):
            Q_old = Q.copy()

            projected = Q @ Z

            # G(y) = log(cosh(y))
            # g(y) = tanh(y)
            # g'(y) = 1 - tanh^2(y)

            g = np.tanh(projected)
            g_prime = 1.0 - g ** 2

            # Fixed point
            first_term = g @ Z.T / n_samples
            second_term = np.mean(g_prime, axis=1)[:, None] * Q
            Q = first_term - second_term

            # Symmetric orthogonalization
            Q = self._symmetric_orthogonalization(Q)

            # Convergence
            similarities = np.abs(np.sum(Q * Q_old, axis=1))
            convergence_value = np.max(1.0 - similarities)

            if convergence_value < self.tolerance:
                converged = True
                break

        if not converged:
            print("not converged")

        S = Q @ Z

        return FastICAResult(
            unmixing=Q,
            sources=S,
            n_iter=iteration,
            converged=converged,
            convergence_value=convergence_value
        )

    def _symmetric_orthogonalization(self, Q):
        QQT = Q @ Q.T

        eigenvalues, eigenvectors = np.linalg.eigh(QQT)

        inverse_sqrt = eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :] @ eigenvectors.T
        return inverse_sqrt @ Q
