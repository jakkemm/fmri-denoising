import numpy as np


def remove_linear_trend(Y):
    n_timepoints = Y.shape[0]

    time = np.linspace(-1.0, 1.0, n_timepoints)

    X_trend = np.column_stack([np.ones(n_timepoints), time])

    trend_coef = np.linalg.pinv(X_trend) @ Y
    return Y - X_trend @ trend_coef

def common_candidate_mask(r2_by_method):
    """
    A voxel is used in summary statistics if at least
    one method obtains outer-CV R2 > 0.
    """

    positive_masks = [
        np.isfinite(r2) & (r2 > 0.0)
        for r2 in r2_by_method.values()
    ]
    
    stacked_voxels = np.stack(positive_masks, axis=0)
    return np.any(stacked_voxels, axis=0)

def median_r2_by_method(r2_by_method, candidate_mask):
    if not np.any(candidate_mask):
        return {method: float("nan") for method in r2_by_method}

    medians_by_method = {
        method: np.nanmedian(r2[candidate_mask])
        for method, r2 in r2_by_method.items()
    }
    return medians_by_method

def delta_r2_vs_standard(r2_by_method , standard_name="standard_glm"):
    standard_r2 = r2_by_method[standard_name]

    delta_by_method = {
        method: (r2 - standard_r2)
        for method, r2 in r2_by_method.items()
    }
    return delta_by_method

def _jackknife_beta_statistics(beta_by_fold):
    # beta_by_fold: (n_folds, n_conditions, n_voxels)

    n_folds = beta_by_fold.shape[0]
    
    beta_mean = np.mean(beta_by_fold, axis=0)
    beta_se = np.std(beta_by_fold, axis=0, ddof=0) * np.sqrt(n_folds - 1)
    
    return beta_mean, beta_se

def jackknife_snr_by_method(beta_by_fold_by_method):
    beta_mean_by_method = {}
    beta_se_by_method = {}

    signal_amplitudes = []

    for method, beta_by_fold in beta_by_fold_by_method.items():
        beta_mean, beta_se = _jackknife_beta_statistics(beta_by_fold)

        beta_mean_by_method[method] = beta_mean
        beta_se_by_method[method] = beta_se

        # Magnitude of the largest beta weight per voxel.
        signal_amplitudes.append(np.max(np.abs(beta_mean), axis=0))

    # GLMdenoise DNB averages the signal
    # magnitude across methods before SNR
    # comparison. Therefore differences in SNR
    # reflect stability rather than beta magnitude.
    shared_signal = np.mean(np.stack(signal_amplitudes, axis=0), axis=0,)

    snr_by_method = {}

    for method, beta_se in beta_se_by_method.items():
        noise = np.mean(beta_se, axis=0)

        snr = np.full(noise.shape, np.nan, dtype=np.float32)

        valid = np.isfinite(noise) & (noise > np.finfo(float).eps)
        snr[valid] = shared_signal[valid] / noise[valid]

        snr_by_method[method] = snr

    return beta_mean_by_method, beta_se_by_method, snr_by_method

def normalize_subject_performance(median_r2, standard_name="standard_glm"):
    """
    Standard GLM -> 0
    Best method -> 1

    Equivalent to the normalization used for
    the across-dataset summary in GLMdenoise.
    """

    baseline = median_r2[standard_name]
    best = max(median_r2.values())

    denominator = best - baseline

    if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
        result = {
            method: 0.0 if method == standard_name else float("nan")
            for method in median_r2
        }
        return result

    result = {
        method: (score - baseline) / denominator
        for method, score in median_r2.items()
    }
    return result

def average_normalized_performance(normalized_by_subject):
    methods = normalized_by_subject[0].keys()

    return {
        method: np.nanmean([subject_result[method] for subject_result in normalized_by_subject])
        for method in methods
    }

def summarize_normalized_performance(normalized_by_subject):
    methods = tuple(normalized_by_subject[0].keys())
    mean_performance = {}
    sem_performance = {}

    for method in methods:
        values = np.asarray([
            subject_result[method]
            for subject_result in normalized_by_subject
        ], dtype=float)

        finite_values = values[np.isfinite(values)]

        if finite_values.size == 0:
            mean_performance[method] = np.nan
            sem_performance[method] = np.nan
            continue

        mean_performance[method] = np.mean(finite_values)

        if finite_values.size > 1:
            sem_performance[method] = np.std(finite_values, ddof=1) / np.sqrt(finite_values.size)
        else:
            sem_performance[method] = 0.0

    return mean_performance, sem_performance
