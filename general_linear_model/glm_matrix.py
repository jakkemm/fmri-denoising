import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag
from scipy.signal import fftconvolve
from scipy.stats import gamma

from general_linear_model.constants import CONST, RawData, RunData


def canonical_hrf(time):
    positive_resp = gamma.pdf(time, a=6).astype(np.float32)
    undershoot = gamma.pdf(time, a=16).astype(np.float32)

    hrf = positive_resp - undershoot / np.float32(6.0)
    return (hrf / np.max(hrf)).astype(np.float32)

@dataclass
class GLMMatrixBuilder:
    high_pass_cutoff: float = 128.0    # cutoff for determining number of basis functions
    verbose : bool = True
    
    def build_runs(self, runs: list[RawData], components_by_run=None, n_components=0, sampling_res=0.1) -> list[RunData]:
        
        glm_runs = []
        
        for i, raw_run in enumerate(runs):
            components = None
            
            X_task = self._build_task(raw_run.events_df, sampling_res)
            
            K = math.floor(2 * CONST.run_duration / self.high_pass_cutoff)
            X_drift = self._build_drift(CONST.n_scans, K)
        
            blocks = [X_drift]
            
            if n_components > 0:
                components = components_by_run[i]
                blocks.append(components[:, :n_components])
        
            X_nuisance = np.column_stack(blocks).astype(np.float32)
            X = np.column_stack([X_task, X_nuisance]).astype(np.float32)
        
            n_task = X_task.shape[1]
            n_drift = X_drift.shape[1]
            n_total = X.shape[1]
            ts, ds, ps = self._build_slices(n_task, n_drift, n_total)
        
            glm_runs.append(
                RunData(
                    Y=raw_run.Y,
                    X=X,
                    task_slice=ts,
                    drift_slice=ds,
                    pca_slice=ps
                )
            )
            
        return glm_runs
    
    def combine(self, runs: list[RunData]):
        has_pca = runs[0].pca_slice is not None
        
        Y = np.vstack([run.Y for run in runs]).astype(np.float32)
        X_task = np.vstack([run.X[:, run.task_slice] for run in runs]).astype(np.float32)
        X_drift = block_diag(*[run.X[:, run.drift_slice] for run in runs]).astype(np.float32)
        
        design_blocks = [X_task, X_drift]
        
        if has_pca:
            X_pca = block_diag(*[run.X[:, run.pca_slice] for run in runs])
            design_blocks.append(X_pca)
        
        X = np.column_stack(design_blocks).astype(np.float32)
        
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
    def _build_drift(n_scans, K):
        dct_columns = []
        scan_indices = np.arange(n_scans, dtype=np.float32)
        
        dct_columns = [
            np.cos(np.pi * k * (2 * scan_indices + 1) / (2 * n_scans))
            for k in range(K + 1)
        ]

        return np.column_stack(dct_columns).astype(np.float32)
        
    @staticmethod
    def _build_task(events_df, sampling_res):
        frame_times = np.arange(CONST.n_scans, dtype=np.float32) * CONST.tr

        high_res_times = np.arange(0, CONST.run_duration + CONST.tr, sampling_res, dtype=np.float32)
        
        stimulus = GLMMatrixBuilder._build_stimulus(events_df, high_res_times)
        
        hrf_times = np.arange(0.0, 32.0 + sampling_res, sampling_res, dtype=np.float32)
        hrf = canonical_hrf(hrf_times)

        predicted_bold = np.column_stack([
            fftconvolve(
                stimulus[:, i],
                hrf,
                mode="full",
            )[:len(high_res_times)] * sampling_res
            for i in range(stimulus.shape[1])
        ]).astype(np.float32)
        
        frame_indices = np.rint(frame_times / sampling_res).astype(int)

        return predicted_bold[frame_indices]
    
    @staticmethod
    def _build_stimulus(events_df, high_res_times):
        activations = []
        
        for cat in CONST.categories:
            cat_mask = events_df["trial_type"] == cat
            
            onsets = events_df.loc[cat_mask, "onset"].to_numpy()
            durations = events_df.loc[cat_mask, "duration"].to_numpy()
            activation = np.zeros_like(high_res_times, dtype=np.float32)
            
            for start, duration in zip(onsets, durations):
                mask = (high_res_times >= start) & (high_res_times < start + duration)
                activation[mask] = 1.0

            activations.append(activation)
        
        return np.column_stack(activations).astype(np.float32)
