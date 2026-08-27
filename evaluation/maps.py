import numpy as np
from nilearn.masking import unmask

from utils.constants import CONST


def beta_to_img(task_coef, category, mask_img):
    try:
        category_index = CONST.categories.index(category)
    except ValueError:
        raise ValueError(f"Unknown category: {category}")

    beta_values = task_coef[category_index]

    return unmask(beta_values, mask_img)

def pairwise_contrast(positive_category, negative_category):
    contrast = np.zeros(CONST.n_categories, dtype=float)

    try:
        positive_index = CONST.categories.index(positive_category)
        negative_index = CONST.categories.index(negative_category)

    except ValueError as error:
        raise ValueError(f"Unknown task category: {error}")

    contrast[positive_index] = 1.0
    contrast[negative_index] = -1.0

    return contrast

def calculate_contrast_t(task_coef, task_covariance_base, residual_variance, contrast):
    contrast = np.asarray(contrast, dtype=float)

    if contrast.shape[0] != task_coef.shape[0]:
        raise ValueError("Contrast length must equal the number of task regressors.")

    effect = contrast @ task_coef
    variance_scale = contrast @ task_covariance_base @ contrast

    if variance_scale <= 0:
        raise ValueError("Contrast variance is not positive.")

    standard_error = np.sqrt(residual_variance * variance_scale)

    t_values = np.full(effect.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(standard_error) & (standard_error > np.finfo(float).eps)
    t_values[valid] = effect[valid] / standard_error[valid]

    return effect, standard_error, t_values

def contrast_t_to_img(task_coef, task_covariance_base, residual_variance, contrast, mask_img):
    _, _, t_values = calculate_contrast_t(
        task_coef=task_coef,
        task_covariance_base=task_covariance_base,
        residual_variance=residual_variance,
        contrast=contrast
    )

    return unmask(t_values, mask_img)
