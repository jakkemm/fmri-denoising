import math
from dataclasses import dataclass

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

@dataclass
class GLMMatrixBuilder:
    high_pass_cutoff: float = 128.0    # cutoff for determining number of basis functions
    verbose : bool = True
    
    def build(self, run: RawData, components=None, n_components=0, sampling_res=0.1):
        X_task = self.build_task_per_run(run.events_df, sampling_res)
        
        K = math.floor(2 * CONST.run_duration / self.high_pass_cutoff)
        X_drift = self.build_drift(CONST.n_scans, K)
        
        blocks = [X_drift]
            
        if n_components > 0:
            if components is None:
                raise ValueError("PCA components are required when n_components is positive")
            
            blocks.append(components[:, :n_components])
        
        X_nuisance = np.column_stack(blocks)
        X = np.column_stack([X_task, X_nuisance])
        
        n_task = X_task.shape[1]
        n_drift = X_drift.shape[1]
        n_total = X.shape[1]
        ts, ds, ps = self._build_slices(n_task, n_drift, n_total)
        
        return RunData(
            Y=run.Y,
            X=X,
            task_slice=ts,
            drift_slice=ds,
            pca_slice=ps
        )
    
    def combine(self, runs: list[RunData]):
        has_pca = runs[0].pca_slice is not None
        
        Y = np.vstack([run.Y for run in runs])
        X_task = np.vstack([run.X[:, run.task_slice] for run in runs])
        X_drift = block_diag(*[run.X[:, run.drift_slice] for run in runs])
        
        design_blocks = [X_task, X_drift]
        
        
        if has_pca:
            X_pca = block_diag(*[run.X[:, run.pca_slice] for run in runs])
            design_blocks.append(X_pca)
        
        X = np.column_stack(design_blocks)
        
        n_task = X_task.shape[1]
        n_drift = X_drift.shape[1]
        n_total = X.shape[1]
        ts, ds, ps = self._build_slices(n_task, n_drift, n_total)
        
        return RunData(
            Y=Y,
            X=X,
            task_slice=ts,
            drift_slice=ds,
            pca_slice=ps
        )
    
    @staticmethod
    def _build_slices(n_task, n_drift, n_total):
        task_slice = slice(0, n_task)
        drift_slice = slice(n_task, n_task+n_drift)
        pca_slice = slice(n_task+n_drift, n_total) if n_total != n_drift + n_task else None
        
        return task_slice, drift_slice, pca_slice
    
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
            fftconvolve(
                stimulus[:, i],
                hrf,
                mode="full",
            )[:len(high_res_times)] * sampling_res
            for i in range(stimulus.shape[1])
        ])
        
        frame_indices = np.rint(frame_times / sampling_res).astype(int)

        return predicted_bold[frame_indices]
    
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
