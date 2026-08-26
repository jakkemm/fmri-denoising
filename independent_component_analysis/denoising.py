from dataclasses import dataclass
from time import perf_counter

import numpy as np

from independent_component_analysis.classification import ICAComponentClassifier
from independent_component_analysis.fastica import SymmetricFastICA
from independent_component_analysis.model_order_selection import ModelOrderSelection
from utils.constants import BoolArray, FloatArray, RunData
from utils.misc import log
from utils.prewhitening import create_prewhitening_matrix


def lag1_corr(Y):
    Y = Y - Y.mean(axis=0, keepdims=True)

    numerator = np.sum(Y[:-1] * Y[1:], axis=0)
    denominator = np.sqrt(
        np.sum(Y[:-1] ** 2, axis=0)
        * np.sum(Y[1:] ** 2, axis=0)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        rho = numerator / denominator

    return rho

@dataclass
class ICARunDenoisingResult:
    Y_denoised: FloatArray
    q_hat: int
    
    task_mask: BoolArray
    nuisance_mask: BoolArray

    r2_actual: FloatArray
    z_scores: FloatArray

    n_iter: int
    converged: bool
    whitening_error: float

class ICADenoiser:
    def __init__(
        self,
        threshold=0.0,
        n_permutations=1000,
        tolerance=1e-5,
        max_iter=1000,
        random_state=0,
        chunk_size=5000,
        verbose=True,
    ):
        self.chunk_size = chunk_size
        self.verbose = verbose

        self.model_order = ModelOrderSelection()
        self.fastica = SymmetricFastICA(tolerance=tolerance, max_iter=max_iter)
        self.classifier = ICAComponentClassifier(
            threshold=threshold,
            n_permutations=n_permutations,
            random_state=random_state,
        )

    def fit_transform(self, run: RunData) -> ICARunDenoisingResult:
        total_start = perf_counter()
        
        Y = run.Y
        X = run.X

        self._log(f"Starting ICA for {Y.shape[0]} time points and {Y.shape[1]} voxels.")

        V = create_prewhitening_matrix(X=X, Y=Y, chunk_size=self.chunk_size)

        # V = L L^T
        L = np.linalg.cholesky(V, upper=False)

        # Y_pw = L^{-1} Y is the same as
        # L Y_pw = Y, hence np.linalg.solve
        Y_pw = np.linalg.solve(L, Y)

        # TODO: write about standardizing in the thesis and its consequences
        # mainly how centering means, that each voxel (column) will be orthogonal to the vector of ones
        # since sum of each element in column will be equal to 0
        voxel_std = Y_pw.std(axis=0, keepdims=True)
        valid = voxel_std[0] > 1e-8

        X_ica = Y_pw[:, valid]
        X_ica = X_ica - X_ica.mean(axis=0, keepdims=True)
        X_ica = X_ica / X_ica.std(axis=0, keepdims=True)

        selection = self.model_order.find_best_q(X_ica)
        
        q_hat = selection.q_hat
        U_q = selection.eigenvectors[:, :q_hat]
        lambda_q = selection.eigenvalues[:q_hat]

        # Dimensionality reduction
        X_q = U_q.T @ X_ica

        # ICA Whitening
        whitening_matrix = np.diag(1 / np.sqrt(lambda_q))
        Z = whitening_matrix @ X_q

        # For debugging
        # ZZ^T / v should be approximately I
        covariance_Z = (Z @ Z.T) / Z.shape[1]
        whitening_error = np.max(np.abs(covariance_Z - np.eye(q_hat)))

        self._log(
            f"Selected "
            f"q={q_hat}; "
            f"whitening error="
            f"{whitening_error:.3e}"
        )

        # FastICA
        fastica_result = self.fastica.fit_transform(Z)
        Q = fastica_result.unmixing
        S = fastica_result.sources

        # Mixing Matrix
        A = U_q @ (np.diag(np.sqrt(lambda_q))) @ Q.T

        # Classify Components
        X_task = X[:, run.task_slice]

        A_original = L @ A
        classification = self.classifier.classify(
            A=A_original,
            X_task=X_task,
        )
        nuisance_mask = classification.nuisance_mask

        self._log(
            f"Task components: "
            f"{np.sum(classification.task_mask)} | "
            f"Nuisance components: "
            f"{np.sum(nuisance_mask)}"
        )

        # Reconstruct nuisance
        # Y_nuis = A_N S_N
        if np.any(nuisance_mask):
            Y_nuisance = A[:, nuisance_mask] @ S[nuisance_mask, :]
        else:
            Y_nuisance = np.zeros_like(Y, dtype=float)

        # Remove nuisance from original
        Y_denoised = Y - Y_nuisance

        self._log(
            f"Finished ICA | "
            f"q={q_hat} | "
            f"FastICA converged="
            f"{fastica_result.converged} | "
            f"iterations="
            f"{fastica_result.n_iter} | "
            f"time="
            f"{perf_counter() - total_start:.3f}s"
        )

        return ICARunDenoisingResult(
            Y_denoised=Y_denoised,
            q_hat=q_hat,
            task_mask=classification.task_mask,
            nuisance_mask=classification.nuisance_mask,
            r2_actual=classification.r2_actual,
            z_scores=classification.z_scores,
            n_iter=fastica_result.n_iter,
            converged=fastica_result.converged,
            whitening_error=whitening_error
        )

    def _log(self, message):
        if self.verbose:
            log(module="ICA", message=message)
