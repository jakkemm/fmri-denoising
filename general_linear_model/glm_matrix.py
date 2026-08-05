import math

import numpy as np
from scipy.linalg import block_diag
from scipy.signal import fftconvolve
from scipy.stats import gamma

from general_linear_model.constants import CONST, RawData, RunData


def canonical_hrf(time):
    positive_resp = gamma.pdf(time, a=6)
    undershoot = gamma.pdf(time, a=16)
    
    hrf = positive_resp - undershoot / 6
    return hrf / np.max(hrf)

class GLMMatrixBuilder:
    high_pass_cutoff = 128.0    # cutoff for determining number of basis functions
    
    def build(self, runs: list[RawData], components_by_run=None, sampling_res=0.1, n_components=0):
        Y = np.vstack([run.Y for run in runs])
        
        X_task = np.vstack([
            self.build_task_per_run(run.events_df, sampling_res)
            for run in runs
        ])
        
        K = math.floor(2 * CONST.run_duration / self.high_pass_cutoff)
        drift_regressors = self.build_drift(CONST.n_scans, K)
        
        nuisance_blocks = []
        
        for run_index, run in enumerate(runs):
            blocks = [drift_regressors]
            
            if n_components > 0:
                if components_by_run is None:
                    raise ValueError("PCA components are required when n_components is positive")
                
                blocks.append(components_by_run[run_index][:, n_components])
            
            nuisance_blocks.append(np.column_stack(blocks))
        
        X_nuisance = block_diag(*nuisance_blocks)
        X = np.column_stack([X_task, X_nuisance])
        
        n_task = X_task.shape[1]
        
        return RunData(
            Y=Y,
            X=X,
            task_slice=slice(0, n_task),
            drift_slice=slice(n_task, X.shape[1] - n_components),
            pca_slice=slice(X.shape[1] - n_components, X.shape[1]) if n_components > 0 else None
        )
    
    @staticmethod
    def build_task_per_run(events_df, sampling_res):
        frame_times = np.arange(CONST.n_scans) * CONST.tr
        task_regressors = GLMMatrixBuilder._build_task(events_df, frame_times, sampling_res)
        return task_regressors
        
    @staticmethod
    def build_drift(n_scans, K):
        dct_columns = []
        scan_indices = np.arange(n_scans)
        
        dct_columns = [
                np.cos(np.pi * k * (2 * scan_indices + 1) / (2 * n_scans))
                for k in range(K + 1)
            ]

        return np.column_stack(dct_columns)
        
    @staticmethod
    def _build_task(events_df, frame_times, sampling_res):
        high_res_times = np.arange(0, CONST.run_duration + CONST.tr, sampling_res)
        
        stimulus = GLMMatrixBuilder._build_stimulus(events_df, high_res_times)
        
        hrf_times = np.arange(0.0, 32.0 + sampling_res, sampling_res)
        hrf = canonical_hrf(hrf_times)

        predicted_bold = np.column_stack([
            fftconvolve(stimulus[:, i], hrf, mode="same")
            for i in range(stimulus.shape[1])
        ])

        return predicted_bold[::int(CONST.tr / sampling_res)]
    
    @staticmethod
    def _build_stimulus(events_df, high_res_times):
        activations = []
        
        for cat in CONST.categories:
            cat_mask = events_df["trial_type"] == cat
            
            onsets = events_df.loc[cat_mask, "onset"].to_numpy()
            durations = events_df.loc[cat_mask, "duration"].to_numpy()
            activation = np.zeros_like(high_res_times)
            
            for start, duration in zip(onsets, durations):
                mask = (high_res_times >= start) & (high_res_times < start + duration)
                activation[mask] = 1.0

            activations.append(activation)
        
        return np.column_stack(activations)
