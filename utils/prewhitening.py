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
    # TODO: descrive the change from global phi over all voxels to median phi
    # this also means describing the choice of not using a signle covariance matrix for each voxel
    # so no \Sigma_i, rather one V
    Xt = X.T
    XtX = X.T @ X

    phi_voxels_by_run = np.empty((n_runs, Y.shape[1]))

    for chunk_slice, Y_chunk in iter_chunks(Y, chunk_size):
        beta_ols_chunk = np.linalg.solve(XtX, Xt @ Y_chunk)
        resid_chunk = Y_chunk - X @ beta_ols_chunk

        resid_blocks = np.split(resid_chunk, n_runs, axis=0)

        for run_index, residuals in enumerate(resid_blocks):
            numerator = np.sum(residuals[:-1] * residuals[1:], axis=0)
            denominator = np.sum(residuals[:-1]**2, axis=0)
            
            phi_voxel = numerator / denominator
            phi_voxels_by_run[run_index, chunk_slice] = phi_voxel
    
    phis = []
    
    for phi_voxel in phi_voxels_by_run:
        phi = np.nanmedian(phi_voxel)
        if np.abs(phi) >= 1:
            raise ValueError("Estimated autoregressive parameter does not satisfy condition `|phi| < 1`.")
    
        phis.append(phi)
            
    return phis

def _create_prewhitening_matrix(phis):
    V_blocks = []
    
    for phi in phis:
        ar1_autocorrelations = phi ** np.arange(CONST.n_scans)
        V_hat = toeplitz(ar1_autocorrelations)
        V_blocks.append(V_hat)
    
    return block_diag(*V_blocks)