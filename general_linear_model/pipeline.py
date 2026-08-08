from dataclasses import dataclass
from time import perf_counter

import numpy as np

from general_linear_model.constants import BoolArray, FloatArray, RawData
from general_linear_model.cv import LeaveOneRunOutEvaluator
from general_linear_model.fgls import FGLSRegressor
from general_linear_model.glm_matrix import GLMMatrixBuilder
from general_linear_model.pca import PCADriftRegressorExtractor
from utils.misc import log


@dataclass
class PCADenoisingResult:
    selected_n_components: int
    noise_mask: BoolArray
    cv_scores: dict[int, float]
    coef: FloatArray
    task_coef: FloatArray
    components_by_run: list[FloatArray]
    

class PCADenoisingPipeline:
    def __init__(
            self, 
            k_max : int = 20, 
            high_pass_cutoff : float = 128.0,
            chunk_size : int = 5000, 
            verbose : bool = True, 
        ):
        self.k_max = k_max
        self.verbose = verbose

        self.cv = LeaveOneRunOutEvaluator(chunk_size=chunk_size, verbose=verbose)
        self.glm_builder = GLMMatrixBuilder(high_pass_cutoff=high_pass_cutoff, verbose=verbose)
        self.pca_extractor = PCADriftRegressorExtractor(chunk_size=chunk_size, verbose=verbose)

    def fit(self, runs: list[RawData]):
        self._log("Starting GLM PCA pipeline")
        start = perf_counter()
        
        # 1. Baseline CV
        glm_runs = self.glm_builder.build_runs(runs)
        baseline_result = self.cv.evaluate(glm_runs=glm_runs, pca_pipeline=False)
        self._log(f"Performed baseline cross-validation in {perf_counter() - start:.3f} seconds.")

        # 2. Noise-pool selection
        t = perf_counter()
        noise_mask = baseline_result.noise_pool_mask

        if not np.any(noise_mask):
            raise RuntimeError("The noise pool is empty.")

        # 3. PCA noise regressors
        components_by_run = [
            self.pca_extractor.fit_transform(Y_noise=run.Y[:, noise_mask], P=run.X[:, run.drift_slice])
            for run in glm_runs
        ]

        maximum_available = min(components.shape[1] for components in components_by_run)
        effective_k_max = min(self.k_max, maximum_available)
        
        self._log(f"Extracted components with maximum effective k={effective_k_max} in {perf_counter() - t:.3f} seconds.")

        # 4. Component-count selection
        cv_scores: dict[int, float] = {}

        self._log("Starting cross-validation components evaluation...")
        
        components_time = 0.0
        r2_by_k = {}
        
        for k in range(effective_k_max + 1):
            t = perf_counter()
            self._log(f"Evaluating {k} PCA components | k_max={effective_k_max}")
            
            glm_runs_k = self.glm_builder.build_runs(runs, components_by_run, n_components=k)

            result = self.cv.evaluate(
                glm_runs=glm_runs_k,
                pca_pipeline=True
            )

            r2_by_k[k] = result.r2_per_voxel
            
            comp_time = perf_counter() - t
            components_time += comp_time
            self._log(f"Evaluated {k} components in {comp_time:.3f} seconds | R2={result.median_r2:4f} | k_max={effective_k_max}")

        R2 = np.stack([r2_by_k[k] for k in range(effective_k_max + 1)], axis=0)
        candidate_mask = np.any(R2 > 0.0, axis=0)

        cv_scores = {
            k: np.nanmedian(r2_by_k[k][candidate_mask])
            for k in range(effective_k_max + 1)
        }
        selected_k = self._select_component_count(cv_scores)
        
        self._log(
            f"Finished cross-validation of PCA-derived components | "
            f"Best k={selected_k} | "
            f"Maximum k={effective_k_max} | "
            f"Average time per component: {components_time/(effective_k_max+1):.3f} | "
            f"Total time: {perf_counter() - start:.3f}"
        )

        # 5. Final fit
        all_design = self.glm_builder.build_runs(
            runs=runs,
            components_by_run=components_by_run,
            n_components=selected_k,
        )
        final_design = self.glm_builder.combine(runs=all_design)

        Y_all = np.vstack([run.Y for run in glm_runs])

        final_model = FGLSRegressor()
        final_model.fit(X=final_design.X, Y=Y_all)

        task_coef = final_model.coef_[final_design.task_slice]

        return PCADenoisingResult(
            selected_n_components=selected_k,
            noise_mask=noise_mask,
            cv_scores=cv_scores,
            coef=final_model.coef_,
            task_coef=task_coef,
            components_by_run=components_by_run,
        )

    @staticmethod
    def _select_component_count(cv_scores: dict[int, float]):
        baseline = cv_scores[0]

        improvements = {
            k: score - baseline
            for k, score in cv_scores.items()
        }

        maximum_improvement = max(improvements.values())

        if maximum_improvement <= 0.0:
            return 0

        threshold = 0.95 * maximum_improvement

        eligible = [
            k for k, improvement in improvements.items()
            if improvement >= threshold
        ]

        return min(eligible)
    
    def _log(self, message):
        if self.verbose:
            log(module="PIPELINE", message=message)