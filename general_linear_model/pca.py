from dataclasses import dataclass

import numpy as np

from utils.misc import iter_chunks


@dataclass
class PCADriftRegressorExtractor:
    chunk_size = 5_000
    
    def fit_transform(self, Y_noise, P):
        drift_operator = P @ np.linalg.pinv(P.T @ P) @ P.T
        
        temporal_covariance = np.zeros((Y_noise.shape[0], Y_noise.shape[0]), dtype=float)

        for _, Y_chunk in iter_chunks(Y_noise, self.chunk_size):
            Y_detrended = self._remove_drift(Y_noise, drift_operator)
            
            temporal_covariance += Y_detrended @ Y_detrended.T
        
        eigenvalues, U = np.linalg.eigh(temporal_covariance)

        order = np.argsort(eigenvalues)[::-1]
        U = U[:, order]
        return eigenvalues[order]

    def _remove_drift(self, Y, drift_operator):
        return Y - drift_operator @ Y