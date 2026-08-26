from dataclasses import dataclass
from time import perf_counter

import numpy as np

from utils.misc import iter_chunks, log


@dataclass
class PCADriftRegressorExtractor:
    chunk_size: int = 5_000
    verbose : bool = True
    
    def fit_transform(self, Y_noise, P):
        t = perf_counter()
        
        drift_basis = self._construct_drift_basis(P)
        temporal_covariance = np.zeros((Y_noise.shape[0], Y_noise.shape[0]), dtype=np.float32)

        t = perf_counter()
        for _, Y_chunk in iter_chunks(Y_noise, self.chunk_size):
            Y_detrended = self._remove_drift(Y_chunk, drift_basis)
            
            # implementing unit-norm noralization as per GLMdenoise
            # TODO: write about it in the thesis
            norms = np.linalg.norm(Y_detrended, axis=0)
            valid = norms > 1e-8
            Y_detrended[:, valid] /= norms[valid]
            
            temporal_covariance += Y_detrended @ Y_detrended.T
        
        self._log(f"Calculated temporal covariance in {perf_counter() - t:.3f} seconds")
        
        eigenvalues, U = np.linalg.eigh(temporal_covariance)

        order = np.argsort(eigenvalues)[::-1]
        U = U[:, order].astype(np.float32)
        self._log(f"Done decomposing with PCA in {perf_counter() - t:.3f} seconds")
        return U
    
    def out_project_drift(self, Y, P):
        Y_proj = np.empty_like(Y, dtype=np.float32)
        drift_basis = self._construct_drift_basis(P)
        
        for chunk_slice, Y_chunk in iter_chunks(Y, self.chunk_size):
            Y_proj[:, chunk_slice] = self._remove_drift(Y_chunk, drift_basis)
        
        return Y_proj
        
    def _construct_drift_basis(self, P):
        P = np.asarray(P, dtype=np.float32)
        return P @ np.linalg.pinv(P.T @ P) @ P.T
        
    @staticmethod
    def _remove_drift(Y, drift_operator):
        return Y - drift_operator @ Y

    def _log(self, message):
        if self.verbose:
            log(module="PCA", message=message)
