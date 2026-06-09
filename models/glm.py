import numpy as np
import pandas as pd

from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.masking import compute_epi_mask, apply_mask
from nilearn.image import high_variance_confounds, mean_img, concat_imgs

class GLMModel:
    def __init__(self, images, events, t1_img, tr, label, model_name, verbose=False):
        self.images = images
        self.events = events
        self.t1_img = t1_img
        self.tr = tr
        
        self.label = label
        self.model_name = model_name
        self.verbose = verbose
        
        # Future attributes
        self.design_matrices = None
        self.model = None
        
        # Calculable attributes
        self.mean_func = mean_img(concat_imgs(self.images))
        self.mask_img = compute_epi_mask(self.images)
        
        if self.verbose:
            model_name_msg = f": {model_name}" if model_name else ""
            print(f"Initialized new model{model_name_msg}")
    
    def _get_all_task_conditions(self):
        nuisance_prefixes = (
            "constant",
            "drift",
            "poly",
            "hvc_",
        )

        conditions = set()

        for dm in self.design_matrices:
            for col in dm.columns:
                col_lower = col.lower()

                if any(col_lower.startswith(prefix) for prefix in nuisance_prefixes):
                    continue

                if col_lower in {"intercept"}:
                    continue

                conditions.add(col)

        return sorted(conditions)
    
    def make_design_matrices(
        self, hrf_model="spm", use_high_variance_confounds=False, n_confounds=0, drift_model=None
    ):
        """
        Create one design matrix per run.

        GLMdenoise-inspired structure:
            task regressors convolved with HRF
            + optional polynomial drift regressors
            + optional PCA-like nuisance regressors
        """

        self.design_matrices = []
        if self.verbose:
            print("\nMaking design matrices...")
        
        for run_idx, (img, events_df) in enumerate(zip(self.images, self.events), start=1):
            n_scans = img.shape[-1]
            frame_times = np.arange(n_scans) * self.tr

            run_duration_min = (n_scans * self.tr) / 60.0
            drift_order = np.round(run_duration_min / 2.0)
            drift_order = int(run_duration_min / 2.0 + 0.5)     # + 0.5 for rounding up

            add_regs = None
            add_reg_names = None

            if use_high_variance_confounds and n_confounds > 0:
                confounds = high_variance_confounds(
                    img,
                    n_confounds=n_confounds,
                    detrend=True,
                )

                add_regs = confounds
                add_reg_names = [f"hvc_{i + 1}" for i in range(n_confounds)]

            design_matrix = make_first_level_design_matrix(
                frame_times=frame_times,
                events=events_df,
                hrf_model=hrf_model,
                drift_model=drift_model,
                drift_order=drift_order,
                add_regs=add_regs,
                add_reg_names=add_reg_names,
                min_onset=0,
            )

            if self.verbose:
                verbose_msg = (
                    f"Run {run_idx:02d}: "
                    f"n_scans={n_scans}, "
                    f"duration={run_duration_min:.2f} min, "
                    f"drift_order={drift_order}, "
                    f"design shape={design_matrix.shape}"
                )
                print(verbose_msg)

            self.design_matrices.append(design_matrix)

        self.conditions_names = self._get_all_task_conditions()
        
        return self
    
    def fit(self, noise_model="ar1"):
        if self.verbose:
            print("\nFitting model...")
        self.model = FirstLevelModel(
            noise_model=noise_model,
            standardize=False,
            signal_scaling=0,
            minimize_memory=False,
            subject_label=self.label
        )
        
        self.model = self.model.fit(
            run_imgs=self.images,
            design_matrices=self.design_matrices,
        )
        
        return self
    
    def _make_condition_contrasts(self, condition_name):
        """
        Creates contrast condition_name > baseline
        
        Returns one contrast vector per run.
        """
        contrasts = []
        
        for dm in self.design_matrices:
            contrast = np.zeros(dm.shape[1])
            
            if condition_name in dm.columns:
                contrast[dm.columns.get_loc(condition_name)] = 1.0
            
            contrasts.append(contrast)
        
        return contrasts
    
    def compute_contrast_metrics_for_condition(self, stat_type, output_type, threshold=None):
        rows = []
        
        if self.verbose:
            verbose_msg = (
                f"Calculating metrics: "
                f"statistic={stat_type}, "
                f"output={output_type}, "
                f"threshold={threshold:.2f}"
            )
            print(verbose_msg)
            
        for condition in self.conditions_names:
            contrast = self._make_condition_contrasts(condition)
            
            img = self.model.compute_contrast(
                contrast,
                stat_type=stat_type,
                output_type=output_type
            )
            
            values = apply_mask(img, self.mask_img)
            values = values[np.isfinite(values)]
            
            row = {
                "model": self.model_name,
                "condition": condition,
                "stat_type": stat_type,
                "output_type": output_type,
                "mean": np.mean(values),
                "mean_abs": np.mean(np.abs(values)),
                "max": np.max(values),
                "min": np.min(values),
                "median": np.median(values),
            }
            
            if threshold is not None:
                if output_type == "p_value":
                    selected = values < threshold
                else:
                    selected = values > threshold
                    
                row["n_voxels_thresholded"] = np.sum(selected)
                row["percent_voxels_thresholded"] = 100 * np.mean(selected)
                row["threshold"] = threshold

            rows.append(row)

        return pd.DataFrame(rows)
