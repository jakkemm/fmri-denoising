import time
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import block_diag, toeplitz

from general_linear_model.constants import CONST


@dataclass
class FGLSRegressor:
    chunk_size: int = 5_000
    verbose: bool = True
    coef_: np.ndarray | None = field(default=None, init=False)

    def fit(self, X, Y) -> "FGLSRegressor":
        total_start = time.perf_counter()

        X = self._validate_X(X) # (t x p)
        Y = self._validate_Y(Y) # (t x v)
                
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows.")

        n_runs = Y.shape[0] // CONST.n_scans

        # Preliminary OLS and AR(1) estimation, chunked over voxels.
        phis = self._estimate_phis_chunked(X, Y, n_runs)

        # Construction of prewhitening matrix
        start = time.perf_counter()
        V = self._create_prewhitening_matrix(phis)
        self._log(f"V construction finished in {time.perf_counter() - start:.3f} s.")

        V_inv = np.linalg.pinv(V)

        gls_operator = (np.linalg.inv(X.T @ V_inv @ X) @ X.T @ V_inv)

        self.coef_ = self._calculate_coefficients_chunked(gls_operator, Y)

        self._log(f"Total fit time: {time.perf_counter() - total_start:.3f} s")
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("The model must be fitted before prediction.")

        X = self._validate_X(X)
        return X @ self.coef_

    def _estimate_phis_chunked(self, X, Y, n_runs):
        XtX = X.T @ X

        numerators = np.zeros(n_runs, dtype=float)
        denominators = np.zeros(n_runs, dtype=float)

        for _, Y_chunk in self._iter_chunks(Y):
            beta_ols_chunk = np.linalg.solve(XtX, X.T @ Y_chunk)

            resid_chunk = Y_chunk - X @ beta_ols_chunk

            resid_blocks = np.split(resid_chunk, n_runs, axis=0)

            for run_index, block in enumerate(resid_blocks):
                numerators[run_index] += np.sum(block[1:] * block[:-1])
                denominators[run_index] += np.sum(block[:-1] ** 2)

        phis = numerators / denominators

        if np.any(np.abs(phis) >= 1):
            raise ValueError("Estimated autoregressive parameter does not satisfy condition `|phi| < 1`.")

        return phis

    def _calculate_coefficients_chunked(self, gls_operator, Y):
        coefficients = np.empty(
            (gls_operator.shape[0], Y.shape[1]),
            dtype=float,
        )

        for chunk_slice, Y_chunk in self._iter_chunks(Y):
            coefficients[:, chunk_slice] = (gls_operator @ Y_chunk)

        return coefficients

    def _iter_chunks(self, Y):
        for start in range(0, Y.shape[1], self.chunk_size):
            stop = min(start + self.chunk_size, Y.shape[1])

            chunk_slice = slice(start, stop)
            yield chunk_slice, Y[:, chunk_slice]

    def _log(self, message):
        if self.verbose:
            print(f"[FGLS] {message}", flush=True)

    @staticmethod
    def _create_prewhitening_matrix(phis):
        V_blocks = []
        
        for phi in phis:
            ar1_autocorrelations = phi ** np.arange(CONST.n_scans)
            V_hat = toeplitz(ar1_autocorrelations)
            V_blocks.append(V_hat)
        
        return block_diag(*V_blocks)

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        return X
        
    @staticmethod
    def _validate_Y(Y):
        Y = np.asarray(Y, dtype=float)
        
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.ndim != 2:
            raise ValueError("Y must be one- or two-dimensional.")
        
        return Y


if __name__ == "__main__":
    X = np.random.normal(size=(121, 5))
    Y = np.random.normal(size=(121, 10))

    model = FGLSRegressor(
        chunk_size=5,
        verbose=True,
    )
    model.fit(X, Y)

    print(model.coef_)