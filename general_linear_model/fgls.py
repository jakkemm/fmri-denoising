import time
from dataclasses import dataclass, field

import numpy as np

from utils.misc import iter_chunks, log
from utils.prewhitening import create_prewhitening_matrix


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

        V = create_prewhitening_matrix(X, Y, self.chunk_size)
        V_inv = np.linalg.pinv(V)

        gls_operator = (np.linalg.pinv(X.T @ V_inv @ X) @ X.T @ V_inv)

        self.coef_ = self._calculate_coefficients_chunked(gls_operator, Y)

        self._log(f"Total fit time: {time.perf_counter() - total_start:.3f} s")
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("The model must be fitted before prediction.")

        X = self._validate_X(X)
        return X @ self.coef_

    def _calculate_coefficients_chunked(self, gls_operator, Y):
        coefficients = np.empty(
            (gls_operator.shape[0], Y.shape[1]),
            dtype=float,
        )

        for chunk_slice, Y_chunk in iter_chunks(Y, self.chunk_size):
            coefficients[:, chunk_slice] = (gls_operator @ Y_chunk)

        return coefficients

    def _log(self, message):
        if self.verbose:
            log(module="FGLS", message=message)

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