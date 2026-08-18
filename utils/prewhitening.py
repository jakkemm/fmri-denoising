import numpy as np
from scipy.linalg import block_diag, toeplitz

from utils.constants import CONST
from utils.misc import iter_chunks

    
def create_prewhitening_matrix(X, Y, chunk_size):
    n_runs = Y.shape[0] // CONST.n_scans

    # Preliminary OLS and AR(1) estimation, chunked over voxels.
    phis = _estimate_phis_chunked(X, Y, n_runs, chunk_size)

    # Construction of prewhitening matrix
    V = _create_prewhitening_matrix(phis)
    return V

def _estimate_phis_chunked(X, Y, n_runs, chunk_size):
    XtX = X.T @ X

    numerators = np.zeros(n_runs, dtype=float)
    denominators = np.zeros(n_runs, dtype=float)

    for _, Y_chunk in iter_chunks(Y, chunk_size):
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

def _create_prewhitening_matrix(phis):
    V_blocks = []
    
    for phi in phis:
        ar1_autocorrelations = phi ** np.arange(CONST.n_scans)
        V_hat = toeplitz(ar1_autocorrelations)
        V_blocks.append(V_hat)
    
    return block_diag(*V_blocks)