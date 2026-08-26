from dataclasses import dataclass
from time import perf_counter

from general_linear_model.fgls import FGLSRegressor
from general_linear_model.glm_matrix import GLMMatrixBuilder
from independent_component_analysis.denoising import ICADenoiser
from utils.constants import BoolArray, FloatArray, RawData, RunData
from utils.misc import log


@dataclass
class ICADenoisingPipelineResult:
    coef: FloatArray
    task_coef: FloatArray

    task_covariance_base: FloatArray
    residual_variance: FloatArray

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
        self.ica_denoiser = ICADenoiser(
            threshold=threshold,
            n_permutations=n_permutations,
            tolerance=tolerance,
            max_iter=max_iter,
            random_state=random_state,
            chunk_size=chunk_size,
            verbose=verbose,
        )

    def fit(self, runs: list[RawData]):
        total_start = perf_counter()

        self._log("Starting ICA denoising pipeline.")

        glm_runs = self.glm_builder.build_runs(runs)

        ica_results = []
        denoised_runs = []

        for run_index, glm_run in enumerate(glm_runs):
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

        final_data = self.glm_builder.combine(denoised_runs)
        final_model = FGLSRegressor(chunk_size=self.chunk_size, verbose=self.verbose)

        final_model.fit(X=final_data.X, Y=final_data.Y)

        task_coef = final_model.coef_[final_data.task_slice]
        task_covariance_base = final_model.covariance_base_[final_data.task_slice, final_data.task_slice]

        self._log(f"Finished ICA pipeline in {perf_counter() - total_start:.3f}s")

        return ICADenoisingPipelineResult(
            coef=final_model.coef_,
            task_coef=task_coef,
            task_covariance_base=task_covariance_base,
            residual_variance=final_model.residual_variance_,
            q_by_run=[result.q_hat for result in ica_results],
            task_masks_by_run=[result.task_mask for result in ica_results],
            nuisance_masks_by_run=[result.nuisance_mask for result in ica_results],
            z_scores_by_run=[result.z_scores for result in ica_results],
        )

    def _log(self, message):
        if self.verbose:
            log(module="ICA-PIPELINE", message=message)
