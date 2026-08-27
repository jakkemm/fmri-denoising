from dataclasses import dataclass
from time import perf_counter

import numpy as np

from evaluation.benchmark import OuterCVBenchmark
from evaluation.metrics import (
    common_candidate_mask,
    delta_r2_vs_standard,
    jackknife_snr_by_method,
    median_r2_by_method,
    normalize_subject_performance,
)
from general_linear_model.pipeline import PCADenoisingPipeline, StandardGLMPipeline
from independent_component_analysis.pipeline import ICADenoisingPipeline
from utils.constants import RawData
from utils.misc import log


@dataclass
class SubjectEvaluationResult:
    candidate_mask: np.ndarray

    r2_per_voxel: dict[str, np.ndarray]
    median_r2: dict[str, float]
    delta_r2_vs_standard: dict[str, np.ndarray]

    jackknife_beta_mean: dict[str, np.ndarray]
    jackknife_beta_se: dict[str, np.ndarray]
    jackknife_snr: dict[str, np.ndarray]

    final_task_coef: dict[str, np.ndarray]
    final_task_covariance_base: dict[str, np.ndarray]
    final_residual_variance: dict[str, np.ndarray]

    fit_runtime_seconds: dict[str, float]
    outer_cv_runtime_seconds: dict[str, float]
    normalized_performance: dict[str, float]

    final_method_specific_data: dict[str, dict]

class EvaluationPipeline:
    def __init__(
        self,
        k_max=20,
        threshold=0.0,
        n_permutations=1000,
        tolerance=1e-5,
        max_iter=1000,
        high_pass_cutoff=128.0,
        chunk_size=5000,
        random_state=42,
        verbose=True,
    ):
        self.verbose = verbose

        self.benchmark = OuterCVBenchmark(high_pass_cutoff=high_pass_cutoff, chunk_size=chunk_size, verbose=verbose)
        self.method_factories = {
            "standard_glm": lambda: StandardGLMPipeline(
                high_pass_cutoff=high_pass_cutoff,
                chunk_size=chunk_size,
                verbose=verbose,
            ),
            "glm_pca": lambda: PCADenoisingPipeline(
                    k_max=k_max,
                    high_pass_cutoff=high_pass_cutoff,
                    chunk_size=chunk_size,
                    verbose=verbose,
            ),
            "ica": lambda: ICADenoisingPipeline(
                    threshold=threshold,
                    n_permutations=n_permutations,
                    tolerance=tolerance,
                    max_iter=max_iter,
                    high_pass_cutoff=high_pass_cutoff,
                    chunk_size=chunk_size,
                    random_state=random_state,
                    verbose=verbose,
            ),
        }

    def evaluate(self, runs: list[RawData]) -> SubjectEvaluationResult:
        # 1. Outer cross-validation
        outer_results = {}

        for method_name, method_factory in self.method_factories.items():
            self._log(f"Outer-CV: {method_name}")

            outer_results[method_name] = self.benchmark.evaluate(
                runs=runs,
                method_factory=method_factory,
                method_name=method_name,
            )

        r2_by_method = {
            method: result.r2_per_voxel
            for method, result in outer_results.items()
        }

        # 2. Common voxel maxk
        candidate_mask = common_candidate_mask(r2_by_method)

        # 3. Median and Delta R2
        median_r2 = median_r2_by_method(r2_by_method, candidate_mask)
        delta_r2 = delta_r2_vs_standard(r2_by_method)

        # 4. Jackknife SNR
        beta_by_fold_by_method = {
            method: result.beta_by_fold
            for method, result in outer_results.items()
        }
        beta_mean_by_method, beta_se_by_method, snr_by_method = jackknife_snr_by_method(beta_by_fold_by_method)

        # 5. Final full-data fits
        final_task_coef = {}
        final_task_covariance_base = {}
        final_residual_variance = {}
        final_method_specific_data = {}

        fit_runtime_seconds = {}

        for method_name, method_factory in self.method_factories.items():
            self._log(f"Final fit: {method_name}")

            start = perf_counter()

            method = method_factory()
            result = method.fit(runs)

            fit_runtime_seconds[method_name] = perf_counter() - start
            final_task_coef[method_name] = result.task_coef
            final_task_covariance_base[method_name] = result.task_covariance_base
            final_residual_variance[method_name] = result.residual_variance
            final_method_specific_data[method_name] = self._extract_method_specific_data(result, method_name)

        # 6. Normalized performance
        normalized_performance = normalize_subject_performance(median_r2)
        outer_cv_runtime_seconds = {
            method: result.runtime_seconds
            for method, result in outer_results.items()
        }

        return SubjectEvaluationResult(
            candidate_mask=candidate_mask,
            r2_per_voxel=r2_by_method,
            median_r2=median_r2,
            delta_r2_vs_standard=delta_r2,
            jackknife_beta_mean=beta_mean_by_method,
            jackknife_beta_se=beta_se_by_method,
            jackknife_snr=snr_by_method,
            final_task_coef=final_task_coef,
            final_task_covariance_base=final_task_covariance_base,
            final_residual_variance=final_residual_variance,
            fit_runtime_seconds=fit_runtime_seconds,
            outer_cv_runtime_seconds=outer_cv_runtime_seconds,
            normalized_performance=normalized_performance,
            final_method_specific_data=final_method_specific_data
        )

    def _log(self, message):
        if self.verbose:
            log(module="EVALUATION", message=message)
    
    def _extract_method_specific_data(self, result, method_name):
        if method_name == "ica":
            return {
                "q_by_run": result.q_by_run,
                "z_scores_by_run": result.z_scores_by_run,
                "n_task_by_run": [int(mask.sum()) for mask in result.task_masks_by_run],
                "n_nuisance_by_run": [int(mask.sum()) for mask in result.nuisance_masks_by_run],
            }
        elif method_name == "glm_pca":
            return {
                "selected_n_components": result.selected_n_components,
                "noise_mask": result.noise_mask,
                "cv_scores": result.cv_scores
            }

        return {}
