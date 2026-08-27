from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nilearn.plotting import find_cut_slices, plot_stat_map

from evaluation.maps import beta_to_img, contrast_t_to_img

METHOD_ORDER = ("standard_glm", "glm_pca", "ica")
METHOD_LABELS = {
    "standard_glm": "Standard GLM",
    "glm_pca": "GLM + PCA",
    "ica": "ICA",
}


def _save_figure(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")

def _finite_pair(x, y, mask=None):
    x = np.asarray(x)
    y = np.asarray(y)

    valid = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        valid &= mask

    return x[valid], y[valid]

def _square_limits(arrays, padding=0.05):
    values = np.concatenate([
        np.asarray(array)[np.isfinite(array)]
        for array in arrays
    ])

    lower = np.min(values)
    upper = np.max(values)

    if lower == upper:
        return lower - 1.0, upper + 1.0

    span = upper - lower

    return lower - padding * span, upper + padding * span,

def plot_r2_scatter(r2_per_voxel, candidate_mask, output_path=None):
    """
    Standard GLM vs denoising methods.

    R2 is displayed in percent, so differences
    are percentage points.
    """

    standard = r2_per_voxel["standard_glm"] * 100.0
    methods = ("glm_pca", "ica")
    
    compared_arrays = [standard]
    for method in methods:
        compared_arrays.append(r2_per_voxel[method] * 100.0)
    lower, upper = _square_limits(compared_arrays)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    for ax, method in zip(axes, methods):
        method_r2 = (r2_per_voxel[method] * 100.0)
        x, y = _finite_pair(standard, method_r2, candidate_mask)
        
        ax.scatter(x, y, s=7, alpha=0.25)
        ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_xlabel("Standard GLM outer-CV $R^2$ (%)")

        ax.set_ylabel(f"{METHOD_LABELS[method]} " "outer-CV $R^2$ (%)")
        ax.set_title(METHOD_LABELS[method])
        ax.grid(alpha=0.2)

    fig.suptitle("Outer-CV prediction accuracy")
    _save_figure(fig, output_path)
    return fig

def plot_delta_r2_distribution(delta_r2_vs_standard, candidate_mask, output_path=None):
    """
    Distribution of voxelwise improvement over
    Standard GLM.

    Values are percentage points.
    """

    methods = ("glm_pca", "ica")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for ax, method in zip(axes, methods):
        delta = delta_r2_vs_standard[method][candidate_mask] * 100.0
        delta = delta[np.isfinite(delta)]
        median = np.median(delta)

        ax.hist(delta, bins=60)
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.axvline(median, linestyle=":", linewidth=1.5, label=f"median = {median:.4f} pp")
        ax.set_xlabel("$\\Delta R^2$ vs Standard GLM (percentage points)")
        ax.set_ylabel("Number of voxels")
        ax.set_title(METHOD_LABELS[method])
        ax.legend()
        ax.grid(alpha=0.2)

    fig.suptitle("Voxelwise change in prediction accuracy")
    _save_figure(fig, output_path)
    return fig


def plot_binned_r2_improvement(r2_per_voxel, candidate_mask, bin_width_pp=5.0, output_path=None):
    """
    Similar in spirit to Figure 4B of GLMdenoise.

    Voxels are binned according to Standard GLM
    R2 and improvement is summarized inside bins.
    """

    standard = r2_per_voxel["standard_glm"] * 100.0
    methods = ("glm_pca", "ica")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for ax, method in zip(axes, methods):
        method_r2 = r2_per_voxel[method] * 100.0
        
        baseline, denoised = _finite_pair(standard, method_r2, candidate_mask)
        improvement = denoised - baseline

        minimum = np.floor(np.min(baseline) / bin_width_pp) * bin_width_pp
        maximum = np.ceil(np.max(baseline) / bin_width_pp) * bin_width_pp
        bins = np.arange(minimum, maximum + bin_width_pp, bin_width_pp)

        centers = []
        medians = []
        lower_ranges = []
        upper_ranges = []

        for left, right in zip(bins[:-1], bins[1:]):
            mask = (baseline >= left) & (baseline < right)

            # Ignore nearly empty bins.
            if np.sum(mask) < 5:
                continue

            values = improvement[mask]
            median = np.median(values)
            low, high = np.percentile(values, [2.5, 97.5])

            centers.append((left + right) / 2.0)
            medians.append(median)

            lower_ranges.append(median - low)
            upper_ranges.append(high - median)

        centers = np.asarray(centers)
        medians = np.asarray(medians)

        ax.errorbar(centers, medians, yerr=[lower_ranges, upper_ranges], marker="o", linewidth=1, capsize=2)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel(r"Standard GLM outer-CV $R^2$ $(\%)$")
        ax.set_ylabel(r"$\Delta R^2$ (percentage points)")
        ax.set_title(METHOD_LABELS[method])
        ax.grid(alpha=0.2)

    fig.suptitle("Improvement as a function of baseline accuracy")
    _save_figure(fig, output_path)
    return fig

def plot_median_r2(median_r2, output_path=None):
    values = [median_r2[method] * 100.0 for method in METHOD_ORDER]
    labels = [METHOD_LABELS[method] for method in METHOD_ORDER]

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    bars = ax.bar(labels, values)
    ax.set_ylabel(r"Median outer-CV $R^2$ $(\%)$")
    ax.set_title("Median prediction accuracy")
    ax.axhline(0.0, linewidth=1)
    ax.grid(axis="y", alpha=0.2)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            value, 
            f"{value:.4f}", 
            ha="center", 
            va="bottom" if value >= 0 else "top"
        )

    _save_figure(fig, output_path)
    return fig

def plot_snr_scatter(snr_by_method, candidate_mask, output_path=None):
    standard = snr_by_method["standard_glm"]
    methods = ("glm_pca", "ica")

    finite_arrays = [standard[np.isfinite(standard) & candidate_mask]]

    for method in methods:
        values = snr_by_method[method]
        finite_arrays.append(values[np.isfinite(values) & candidate_mask])

    upper = max(np.max(values) for values in finite_arrays if values.size > 0)
    upper *= 1.05

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    for ax, method in zip(axes, methods):
        x, y = _finite_pair(standard, snr_by_method[method], candidate_mask)
        
        ax.scatter(x, y, s=7, alpha=0.25)
        ax.plot([0, upper], [0, upper], linestyle="--", linewidth=1)
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_xlabel("Standard GLM jackknife SNR")
        ax.set_ylabel(f"{METHOD_LABELS[method]} jackknife SNR")
        ax.set_title(METHOD_LABELS[method])
        ax.grid(alpha=0.2)

    fig.suptitle("Stability of task beta estimates")
    _save_figure(fig, output_path)
    return fig

def plot_runtime(fit_runtime_seconds, outer_cv_runtime_seconds, output_path=None):
    labels = [METHOD_LABELS[method] for method in METHOD_ORDER]
    fit_values = [fit_runtime_seconds[method] for method in METHOD_ORDER]
    outer_values = [outer_cv_runtime_seconds[method] for method in METHOD_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    axes[0].bar(labels, fit_values)
    axes[0].set_ylabel("Runtime (s)")
    axes[0].set_title("Full-data fit")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.2)
    
    axes[1].bar(labels, outer_values)
    axes[1].set_ylabel("Runtime (s)")
    axes[1].set_title("Outer-CV benchmark")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("Computational runtime")
    _save_figure(fig, output_path)
    return fig

def _common_abs_vmax(images):
    vmax = max(
        np.nanmax(np.abs(image.get_fdata()))
        for image in images.values()
    )
    
    if not np.isfinite(vmax) or vmax <= 0:
        return 1.0
    return vmax

def _get_cut_coords(reference_img, mask_img, direction, n_cuts):
    try:
        return find_cut_slices(
            reference_img,
            direction=direction,
            n_cuts=n_cuts,
        )
    except Exception:
        # Fallback if the statistical map happens to be nearly empty.
        return find_cut_slices(
            mask_img,
            direction=direction,
            n_cuts=n_cuts,
        )


def _plot_map_comparison(
    images,
    mask_img,
    background_img,
    title,
    threshold,
    output_path=None,
    display_mode="z",
    cut_coords=None,
    n_cuts=5,
):
    reference_img = images["standard_glm"]

    if cut_coords is None:
        cut_coords = _get_cut_coords(reference_img, mask_img, direction=display_mode, n_cuts=n_cuts)

    vmax = _common_abs_vmax(images)

    fig, axes = plt.subplots(len(METHOD_ORDER), 1, figsize=(13, 9))

    for index, method in enumerate(METHOD_ORDER):
        plot_stat_map(
            stat_map_img=images[method],
            bg_img=background_img,
            display_mode=display_mode,
            cut_coords=cut_coords,
            threshold=threshold,
            vmax=vmax,
            symmetric_cbar=True,
            title=METHOD_LABELS[method],
            annotate=True,
            draw_cross=False,
            colorbar=(index == len(METHOD_ORDER) - 1),
            axes=axes[index],
        )

    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(top=0.92, hspace=0.25)
    _save_figure(fig, output_path)
    return fig

def plot_beta_map_comparison(
    result,
    category,
    mask_img,
    t1_img,
    output_path=None,
    display_mode="z",
    cut_coords=None,
    n_cuts=5,
):
    images = {
        method: beta_to_img(
            task_coef=(result.final_task_coef[method]),
            category=category,
            mask_img=mask_img,
        )
        for method in METHOD_ORDER
    }

    return _plot_map_comparison(
        images=images,
        mask_img=mask_img,
        background_img=t1_img,
        title=(f"Beta maps: {category}"),
        threshold=None,     # Do not threshold beta estimates
        output_path=output_path,
        display_mode=display_mode,
        cut_coords=cut_coords,
        n_cuts=n_cuts,
    )

def plot_contrast_t_map_comparison(
    result,
    contrast,
    contrast_name,
    mask_img,
    t1_img,
    threshold=3.0,
    output_path=None,
    display_mode="z",
    cut_coords=None,
    n_cuts=5,
):
    images = {}

    for method in METHOD_ORDER:
        images[method] = contrast_t_to_img(
            task_coef=result.final_task_coef[method],
            task_covariance_base=result.final_task_covariance_base[method],
            residual_variance=result.final_residual_variance[method],
            contrast=contrast,
            mask_img=mask_img,
        )

    return _plot_map_comparison(
        images=images,
        mask_img=mask_img,
        background_img=t1_img,
        title=f"Contrast t-maps: {contrast_name}",
        threshold=threshold,
        output_path=output_path,
        display_mode=display_mode,
        cut_coords=cut_coords,
        n_cuts=n_cuts,
    )

def plot_normalized_performance(mean_performance, sem_performance, output_path=None):
    labels = [METHOD_LABELS[method] for method in METHOD_ORDER]
    means = [mean_performance[method] for method in METHOD_ORDER]
    errors = [sem_performance[method] for method in METHOD_ORDER]

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    ax.bar(labels, means, yerr=errors, capsize=4)
    ax.axhline(0.0, linewidth=1)

    ax.set_ylabel("Normalized performance")
    ax.set_title("Mean normalized performance across subjects")
    ax.grid(axis="y", alpha=0.2)
    
    _save_figure(fig, output_path)
    return fig
