from dataclasses import dataclass
from time import perf_counter

import numpy as np

from evaluation.metrics import remove_linear_trend
from general_linear_model.glm_matrix import GLMMatrixBuilder
from utils.misc import iter_chunks, log


@dataclass
class OuterCVResult:
    r2_per_voxel: np.ndarray
    beta_by_fold: np.ndarray
    runtime_seconds: float


class OuterCVBenchmark:
    def __init__(self, high_pass_cutoff=128.0, chunk_size=5000, verbose=True):
        self.chunk_size = chunk_size
        self.verbose = verbose

        self.glm_builder = GLMMatrixBuilder(high_pass_cutoff=high_pass_cutoff, verbose=False)

    def evaluate(self, runs, method_factory, method_name) -> OuterCVResult:
        total_start = perf_counter()

        n_runs = len(runs)
        n_voxels = runs[0].Y.shape[1]

        sum_y = np.zeros(n_voxels, dtype=np.float64)
        sum_y2 = np.zeros(n_voxels, dtype=np.float64)
        ss_res = np.zeros(n_voxels, dtype=np.float64)

        n_observations = 0

        beta_by_fold = None

        for held_out_index in range(n_runs):
            fold_start = perf_counter()

            self._log(f"{method_name}: outer CV {held_out_index + 1}/{n_runs}")

            training_runs = [
                run
                for index, run in enumerate(runs)
                if index != held_out_index
            ]

            # Fresh method for each fold.
            method = method_factory()
            fit_result = method.fit(training_runs)

            B_task = fit_result.task_coef

            if beta_by_fold is None:
                beta_by_fold = np.empty(
                    (
                        n_runs,
                        B_task.shape[0],
                        B_task.shape[1],
                    ),
                    dtype=np.float32,
                )

            beta_by_fold[held_out_index] = B_task
            held_out_raw = runs[held_out_index]

            held_out_glm = self.glm_builder.build_runs([held_out_raw])[0]

            X_task_test = held_out_glm.X[:, held_out_glm.task_slice]
            Y_pred = X_task_test @ B_task
            Y_true = held_out_raw.Y

            # DNB-style evaluation:
            # remove only constant + linear drift from measured and predicted data.
            Y_true_eval = remove_linear_trend(Y_true)
            Y_pred_eval = remove_linear_trend(Y_pred)

            for (chunk_slice, Y_true_chunk) in iter_chunks(Y_true_eval, self.chunk_size):
                Y_pred_chunk = Y_pred_eval[:, chunk_slice]

                sum_y[chunk_slice] += np.sum(Y_true_chunk, axis=0)
                sum_y2[chunk_slice] += np.sum(Y_true_chunk ** 2, axis=0)
                ss_res[chunk_slice] += np.sum((Y_true_chunk - Y_pred_chunk) ** 2, axis=0)

            n_observations += Y_true_eval.shape[0]

            self._log(f"{method_name}: fold {held_out_index + 1}/{n_runs} finished in {perf_counter() - fold_start:.3f}s")

        ss_tot = sum_y2 - (sum_y ** 2) / n_observations

        r2 = np.full(n_voxels, np.nan, dtype=np.float32)
        valid = ss_tot > np.finfo(float).eps
        r2[valid] = 1.0 - ss_res[valid] / ss_tot[valid]

        return OuterCVResult(
            r2_per_voxel=r2,
            beta_by_fold=beta_by_fold,
            runtime_seconds=(perf_counter() - total_start),
        )

    def _log(self, message):
        if self.verbose:
            log(module="BENCHMARK", message=message)
