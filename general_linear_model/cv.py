from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.metrics import r2_score

from general_linear_model.constants import FloatArray, RawData
from general_linear_model.fgls import FGLSRegressor
from general_linear_model.glm_matrix import GLMMatrixBuilder
from general_linear_model.pca import PCADriftRegressorExtractor
from utils.misc import iter_chunks, log


@dataclass
class NoisePoolResults:
    noise_pool: FloatArray
    r2_per_voxel: FloatArray
    mean_per_voxel: FloatArray
    threshold: float

@dataclass
class PCAComponentsResults:
    pass


class LeaveOneRunOutEvaluator:
    def __init__(self, chunk_size=5000, verbose=True):
        self.verbose = verbose
        self.chunk_size = chunk_size
        
        self.regressor = FGLSRegressor
        self.glm_builder = GLMMatrixBuilder(verbose=self.verbose)
        self.drift_extractor = PCADriftRegressorExtractor(chunk_size=self.chunk_size, verbose=self.verbose)

    def evaluate(self, runs: list[RawData], components_by_run=None, n_components=0):
        total_start = perf_counter()
        
        n_runs = len(runs)
        n_time, n_voxels = runs[0].Y.shape
        
        Y_true_all = np.empty((n_time * n_runs, n_voxels), dtype=float)
        Y_pred_all = np.empty_like(Y_true_all)
        Y_true_raw = np.empty_like(Y_true_all)
        
        
        glm_runs = []
        
        for i, run in enumerate(runs):
            training_components = None
            if components_by_run is not None:
                training_components = components_by_run[i]
            
            glm_run = self.glm_builder.build(
                run=run,
                components=training_components,
                n_components=n_components
            )
            
            glm_runs.append(glm_run)
        
        self._log(f"Prepared data for individual runs in {perf_counter() - total_start:.3f}")

        offset = 0

        for held_out_index in range(len(runs)):
            self._log(f"Calculating run {held_out_index+1}/{len(runs)}")
            held_out_start = perf_counter()
            
            training_indices = [index for index in range(len(runs)) if index != held_out_index]

            training_runs = [glm_runs[index] for index in training_indices]

            training_components = None
            if components_by_run is not None:
                training_components = [components_by_run[index] for index in training_indices]

            glm_data = self.glm_builder.combine(runs=training_runs)

            model = self.regressor(chunk_size=self.chunk_size, verbose=self.verbose)
            model.fit(X=glm_data.X, Y=glm_data.Y)

            B_task = model.coef_[glm_data.task_slice]

            held_out_run = glm_runs[held_out_index]
            Y_pred = held_out_run.X[:, held_out_run.task_slice] @ B_task
            
            Y_raw = held_out_run.Y

            X_drift = held_out_run.X[:, held_out_run.drift_slice]
            Y_true_out = self.drift_extractor.out_project_drift(Y_raw, X_drift)
            Y_pred_out = self.drift_extractor.out_project_drift(Y_pred, X_drift)

            n = Y_true_out.shape[0]
            
            Y_true_all[offset:offset+n] = Y_true_out
            Y_pred_all[offset:offset+n] = Y_pred_out
            Y_true_raw[offset:offset+n] = Y_raw
            
            offset += n
            
            self._log(f"Run {held_out_index+1}/{len(runs)} predicted in {perf_counter() - held_out_start:.3f} seconds")
        
        if n_components == 0:
            CVResults = self.select_noise_pool(Y_true_raw, Y_true_all, Y_pred_all)
        else:
            raise NotImplementedError("Not yet")
        
        self._log(f"Done evaluating in {perf_counter() - total_start:.3f} seconds")
        return CVResults

    
    def select_noise_pool(self, Y_true_raw, Y_true, Y_pred):
        t = perf_counter()
        
        r2_per_voxel = np.empty(Y_true.shape[1], dtype=float)
        mean_per_voxel = np.empty(Y_true.shape[1], dtype=float)
        
        for chunk_slice, Y_true_chunk in iter_chunks(Y_true, self.chunk_size):
            Y_pred_chunk = Y_pred[:, chunk_slice]
            Y_raw_chunk = Y_true_raw[:, chunk_slice]
            
            r2_per_voxel[chunk_slice] = r2_score(Y_true_chunk, Y_pred_chunk, multioutput="raw_values")
            mean_per_voxel[chunk_slice] = np.mean(Y_raw_chunk, axis=0)
            
        threshold = 0.5 * np.percentile(mean_per_voxel, 99)
        noise_pool_mask = (r2_per_voxel < 0) & (mean_per_voxel > threshold)
        
        self._log(f"Calculated metrics and selected {np.sum(noise_pool_mask)} noise voxels for noise pool in {perf_counter() - t:.3f} seconds.")
        
        return NoisePoolResults(
            noise_pool=Y_true_raw[:, noise_pool_mask],
            r2_per_voxel=r2_per_voxel,
            mean_per_voxel=mean_per_voxel,
            threshold=threshold
        )
        
    
    def _log(self, message):
        if self.verbose:
            log(module="CV", message=message)