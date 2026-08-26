from dataclasses import dataclass
from time import perf_counter

import numpy as np

from general_linear_model.fgls import FGLSRegressor
from general_linear_model.glm_matrix import GLMMatrixBuilder
from general_linear_model.pca import PCADriftRegressorExtractor
from independent_component_analysis.denoising import ICADenoiser
from utils.constants import BoolArray, FloatArray, RunData
from utils.misc import iter_chunks, log


@dataclass
class ICACrossValidationResult:
    r2_per_voxel: FloatArray
    median_r2: float


@dataclass
class ICADenoisingPipelineResult:
    cv_r2_per_voxel: FloatArray
    cv_median_r2: float

    coef: FloatArray
    task_coef: FloatArray

    q_by_run: list[int]
    task_masks_by_run: list[BoolArray]
    nuisance_masks_by_run: list[BoolArray]
    z_scores_by_run: list[FloatArray]

class ICADenoisingPipeline:
    def __init__(
        self,
        threshold=0.0,
        n_permutations=1000,
        tolerance=1e-5,
        max_iter=1000,
        high_pass_cutoff=128.0,
        chunk_size=5000,
        random_state=0,
        verbose=True,
    ):

        self.chunk_size = chunk_size
        self.verbose = verbose

        self.glm_builder = GLMMatrixBuilder(high_pass_cutoff=high_pass_cutoff, verbose=verbose)
        self.drift_extractor = PCADriftRegressorExtractor(chunk_size=chunk_size, verbose=verbose)
        self.ica_denoiser = ICADenoiser(
            threshold=threshold,
            n_permutations=n_permutations,
            tolerance=tolerance,
            max_iter=max_iter,
            random_state=random_state,
            chunk_size=chunk_size,
            verbose=verbose,
        )

    def fit(self, runs: list[RunData]) -> ICADenoisingPipelineResult:
        total_start = perf_counter()
        self._log("Starting ICA denoising pipeline.")

        glm_runs = self.glm_builder.build_runs(runs)

        ica_results = []
        denoised_runs = []

        for (run_index, glm_run) in enumerate(glm_runs):
            self._log(f"ICA denoising run {run_index + 1}/{len(glm_runs)}.")
            result = self.ica_denoiser.fit_transform(run=glm_run)

            ica_results.append(result)
            denoised_runs.append(
                RunData(
                    Y=result.Y_denoised,
                    X=glm_run.X,
                    task_slice=glm_run.task_slice,
                    drift_slice=glm_run.drift_slice,
                    pca_slice=glm_run.pca_slice
                )
            )

        cv_result = self._cross_validate(
            raw_runs=glm_runs,
            denoised_runs=denoised_runs
        )

        final_data = self.glm_builder.combine(denoised_runs)
        final_model = FGLSRegressor(chunk_size=self.chunk_size, verbose=self.verbose)

        final_model.fit(X=final_data.X, Y=final_data.Y)

        task_coef = final_model.coef_[final_data.task_slice]

        self._log(
            f"Finished ICA pipeline in "
            f"{perf_counter() - total_start:.3f}s | "
            f"median CV R2="
            f"{cv_result.median_r2:.6f}"
        )

        return ICADenoisingPipelineResult(
                cv_r2_per_voxel=cv_result.r2_per_voxel,
                cv_median_r2=cv_result.median_r2,
                coef=final_model.coef_,
                task_coef=task_coef,
                q_by_run=[result.q_hat for result in ica_results],
                task_masks_by_run=[result.task_mask for result in ica_results],
                nuisance_masks_by_run=[result.nuisance_mask for result in ica_results],
                z_scores_by_run=[result.z_scores for result in ica_results],
        )

    def _cross_validate(
        self, raw_runs: list[RunData], denoised_runs: list[RunData]) -> ICACrossValidationResult:
        n_runs = len(raw_runs)
        n_voxels = (raw_runs[0].Y.shape[1])

        sum_y = np.zeros(n_voxels, dtype=np.float64)
        sum_y2 = np.zeros(n_voxels, dtype=np.float64)
        ss_res = np.zeros(n_voxels, dtype=np.float64)

        n_observations = 0

        for held_out_index in range(n_runs):
            fold_start = perf_counter()

            self._log(
                f"ICA CV fold "
                f"{held_out_index + 1}"
                f"/{n_runs}"
            )

            training_runs = [
                denoised_runs[index]
                for index in range(n_runs)
                if index != held_out_index
            ]
            training_data = self.glm_builder.combine(training_runs)

            model = FGLSRegressor(chunk_size=self.chunk_size, verbose=self.verbose)

            model.fit(X=training_data.X, Y=training_data.Y)
            B_task = model.coef_[training_data.task_slice]

            held_out = raw_runs[held_out_index]

            X_task_test = held_out.X[:, held_out.task_slice]
            Y_pred = X_task_test @ B_task

            Y_true = held_out.Y
            X_drift = held_out.X[:, held_out.drift_slice]

            Y_true_out = self.drift_extractor.out_project_drift(Y_true, X_drift)
            Y_pred_out = self.drift_extractor.out_project_drift(Y_pred, X_drift)

            for (chunk_slice, Y_true_chunk) in iter_chunks(Y_true_out, self.chunk_size):
                Y_pred_chunk = Y_pred_out[:, chunk_slice]

                sum_y[chunk_slice] += np.sum(Y_true_chunk, axis=0)
                sum_y2[chunk_slice] += np.sum(Y_true_chunk ** 2, axis=0)

                ss_res[chunk_slice] += np.sum((Y_true_chunk - Y_pred_chunk) ** 2, axis=0)

            n_observations += Y_true_out.shape[0]

            self._log(
                f"Finished fold "
                f"{held_out_index + 1}"
                f"/{n_runs} in "
                f"{perf_counter() - fold_start:.3f}s"
            )

        ss_tot = sum_y2 - (sum_y ** 2) / n_observations

        r2 = np.full(n_voxels, np.nan, dtype=np.float32)
        valid = ss_tot > np.finfo(float).eps
        r2[valid] = 1.0 - ss_res[valid] / ss_tot[valid]

        candidate_mask = np.isfinite(r2) & (r2 > 0.0)

        if np.any(candidate_mask):
            median_r2 = np.median(r2[candidate_mask])
        else:
            median_r2 = float("nan")

        return ICACrossValidationResult(
            r2_per_voxel=r2,
            median_r2=median_r2,
        )

    def _log(self, message):
        if self.verbose:
            log(module="ICA-PIPELINE", message=message)
