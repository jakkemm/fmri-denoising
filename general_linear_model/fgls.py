from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag, fractional_matrix_power, toeplitz

from GeneralLinearModel.constants import CONST


@dataclass
class FGLSRegressor:
    coef_ = None 
    
    def fit(self, X, Y, n_runs=None) -> "FGLSRegressor":
        X = self._validate_X(X) # (t x p)
        Y = self._validate_Y(Y) # (t x v)
    
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows.")
        
        beta_ols = np.linalg.pinv(X.T @ X) @ X.T @ Y    # (p x v)
        resid = Y - X @ beta_ols                        # (t x v)
        print("Calculated OLS residuals.")
        
        phis = self._estimate_phis(resid, n_runs)
        W = self._create_prewhitening_matrix(phis)
        print("Created matrix for prewhitening.")
        
        W_inv = np.linalg.pinv(W)
        self.coef_ = np.linalg.inv(X.T @ W_inv @ X) @ X.T @ W_inv @ Y
        print("Calculated FGLS coefficients.")
        return self
    
    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("The model must be fitted before prediction.")
        
        X = self._validate_X(X)
        return X @ self.coef_
    
    @staticmethod
    def _create_prewhitening_matrix(phis):
        W_blocks = []
        
        for phi in phis:
            ar1_autocorrelations = phi ** np.arange(CONST.n_scans)
            V_hat = toeplitz(ar1_autocorrelations)
            W = fractional_matrix_power(V_hat, -0.5)
            W_blocks.append(W)
        
        return block_diag(*W_blocks)
    
    @staticmethod
    def _estimate_phis(resid, n_runs):
        resid_blocks = np.split(resid, n_runs) if n_runs else [resid]
        phis = []
        
        for block in resid_blocks:
            num = float(np.sum(block[1:] * block[:-1]))
            denom = float(np.sum(block[:-1] ** 2))
            
            phi = num / denom
            if np.abs(phi) >= 1:
                raise ValueError("Estimated autoregressive parameter does not satisfy condition `|phi| < 1`.")
            
            phis.append(phi)
        
        return phis
    
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
    
    model = FGLSRegressor()
    model.fit(X, Y)
    print(model.coef_)
