from dataclasses import dataclass

import numpy as np

from utils.constants import BoolArray, FloatArray
from utils.misc import calculate_r2


@dataclass
class ICAClassificationResult:
    task_mask: BoolArray
    nuisance_mask: BoolArray

    r2_actual: FloatArray
    shuffled_mean: FloatArray
    shuffled_std: FloatArray
    z_scores: FloatArray


class ICAComponentClassifier:
    def __init__(self, threshold=0.0, n_permutations=1000, random_state=42):
        self.threshold = float(threshold)
        self.n_permutations = int(n_permutations)

        self.rng = np.random.default_rng(random_state)

    def classify(self, A: FloatArray, X_task: FloatArray) -> ICAClassificationResult:
        # Center response and task regressors
        # Equivalent of an intercept, which is missing in the thesis
        A_centered = A - A.mean(axis=0, keepdims=True)
        X_centered = X_task - X_task.mean(axis=0, keepdims=True)

        # Task projection
        XtX = X_centered.T @ X_centered
        task_operator = np.linalg.pinv(XtX) @ X_centered.T

        # Actual task-explained R^2.
        gamma = task_operator @ A_centered
        A_fitted = X_centered @ gamma

        r2_actual = calculate_r2(A_centered, A_fitted)

        n_timepoints, n_components = A.shape

        shuffled_mean = np.full(n_components, np.nan, dtype=float)
        shuffled_std = np.full(n_components, np.nan, dtype=float)
        z_scores = np.full(n_components, np.nan, dtype=float)

        permutation_indices = np.array([
            self.rng.permutation(n_timepoints)
            for _ in range(self.n_permutations)
        ])

        eps = np.finfo(float).eps

        for component_index in range(n_components):
            a = A_centered[:, component_index]

            # (B, t) -> (t, B)
            shuffled = a[permutation_indices].T

            gamma_shuffled = task_operator @ shuffled
            shuffled_fitted = X_centered @ gamma_shuffled

            shuffled_r2 = calculate_r2(shuffled, shuffled_fitted)

            mu = np.nanmean(shuffled_r2)
            sigma = np.nanstd(shuffled_r2, ddof=1)

            shuffled_mean[component_index] = mu
            shuffled_std[component_index] = sigma

            actual = r2_actual[component_index]

            if np.isfinite(actual) and np.isfinite(mu) and np.isfinite(sigma) and sigma > eps:
                z_scores[component_index] = (actual - mu) / sigma

        # Conservative behaviour:
        # invalid/undefined components are retained.
        nuisance_mask = np.isfinite(z_scores) & (z_scores <= self.threshold)
        task_mask = ~nuisance_mask

        return ICAClassificationResult(
            task_mask=task_mask,
            nuisance_mask=nuisance_mask,
            r2_actual=r2_actual,
            shuffled_mean=shuffled_mean,
            shuffled_std=shuffled_std,
            z_scores=z_scores,
        )
