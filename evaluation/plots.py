import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from nilearn.plotting import find_cut_slices, plot_stat_map

from evaluation.maps import beta_to_img, contrast_t_to_img

METHOD_ORDER = ("standard_glm", "glm_pca", "ica")
METHOD_LABELS = {
    "standard_glm": "Standard GLM",
    "glm_pca": "GLM + PCA",
    "ica": "ICA",
}

sns.set_theme()

def _save_figure(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())

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

def plot_r2_scatter(r2_per_voxel_sub, candidate_mask_sub, method, output_path=None):
    subjects = list(r2_per_voxel_sub.keys())

    fig, axes = plt.subplots(2, 3, figsize=(10, 7), constrained_layout=True)
    axes = axes.ravel()

    compared_arrays = []

    for subject in subjects:
        standard = r2_per_voxel_sub[subject]["standard_glm"] * 100.0
        method_r2 = r2_per_voxel_sub[subject][method] * 100.0

        mask = candidate_mask_sub[subject]
        x, y = _finite_pair(standard, method_r2, mask)
        compared_arrays.extend([x, y])

    lower, upper = _square_limits(compared_arrays)

    for ax, subject in zip(axes, subjects):
        standard = r2_per_voxel_sub[subject]["standard_glm"] * 100.0
        method_r2 = r2_per_voxel_sub[subject][method] * 100.0

        x, y = _finite_pair(standard, method_r2, candidate_mask_sub[subject])

        ax.scatter(x, y, s=7, alpha=0.25)
        ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(subject)
        ax.grid(alpha=0.2)

    # Hide unused panels if ever needed.
    for ax in axes[len(subjects):]:
        ax.set_visible(False)
    for ax in axes[3:]:
        ax.set_xlabel(r"Standard GLM outer-CV $R^2$ $(\%)$")
    for ax in axes[::3]:
        ax.set_ylabel(rf"{METHOD_LABELS[method]} outer-CV $R^2$ $(\%)$")

    fig.suptitle(f"Outer-CV prediction accuracy: {METHOD_LABELS[method]}")
    _save_figure(fig, output_path)
    return fig

def plot_delta_r2_distribution(delta_r2_vs_standard_sub, candidate_mask_sub, method, output_path=None):
    subjects = list(delta_r2_vs_standard_sub.keys())

    fig, axes = plt.subplots(2, 3, figsize=(10, 7), constrained_layout=True, sharey=True)
    axes = axes.ravel()

    for ax, subject in zip(axes, subjects):
        delta = delta_r2_vs_standard_sub[subject][method][candidate_mask_sub[subject]] * 100.0

        delta = delta[np.isfinite(delta)]
        median = np.median(delta)

        ax.hist(delta, bins="sqrt", color="C0", alpha=0.85)
        ax.axvline(0.0, linestyle="--", linewidth=1, color="black")
        ax.axvline(median, linestyle=":", linewidth=1.5, color="C0", label=f"median = {median:.4f} pp")

        ax.set_title(subject)
        ax.grid(alpha=0.2)
        ax.legend()

    for ax in axes[::3]:
        ax.set_ylabel("Number of voxels")
    for ax in axes[3:]:
        ax.set_xlabel(r"$\Delta R^2$ vs Standard GLM (pp)")

    y_max = max(ax.get_ylim()[1] for ax in axes)    
    for ax in axes:
        ax.set_ylim(0, y_max)

    fig.suptitle(f"Voxelwise change in prediction accuracy: {METHOD_LABELS[method]}")
    _save_figure(fig, output_path)
    return fig

def plot_binned_r2_improvement(r2_per_voxel_sub, candidate_mask_sub, method, bin_width_pp=10.0, output_path=None):
    subjects = list(r2_per_voxel_sub.keys())
    n_subjects = len(subjects)

    all_baseline = []

    for subject in subjects:
        standard = r2_per_voxel_sub[subject]["standard_glm"] * 100.0
        method_r2 = r2_per_voxel_sub[subject][method] * 100.0
        candidate_mask = candidate_mask_sub[subject]

        baseline, _ = _finite_pair(standard, method_r2, candidate_mask)
        if baseline.size > 0:
            all_baseline.append(baseline)

    all_baseline = np.concatenate(all_baseline)
    minimum = np.floor(np.min(all_baseline) / bin_width_pp) * bin_width_pp
    maximum = np.ceil(np.max(all_baseline) / bin_width_pp) * bin_width_pp

    bins = np.arange(minimum, maximum + bin_width_pp, bin_width_pp)
    centers = (bins[:-1] + bins[1:]) / 2.0

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    group_width = 0.5 * bin_width_pp
    offsets = np.linspace(-group_width / 2, group_width / 2, n_subjects)
    colors = plt.cm.Dark2(np.arange(n_subjects))

    for subject_index, subject in enumerate(subjects):
        standard = r2_per_voxel_sub[subject]["standard_glm"] * 100.0
        method_r2 = r2_per_voxel_sub[subject][method] * 100.0
        candidate_mask = candidate_mask_sub[subject]

        baseline, denoised = _finite_pair(standard, method_r2, candidate_mask)
        improvement = denoised - baseline
        plotted_label = False

        for bin_index, (left, right) in enumerate(itertools.pairwise(bins)):
            mask = (baseline >= left) & (baseline < right)

            if np.sum(mask) < 5:
                continue

            values = improvement[mask]

            median = np.median(values)
            low, high = np.percentile(values, [2.5, 97.5])

            x = centers[bin_index] + offsets[subject_index]

            ax.vlines(x, low, high, linewidth=5, color=colors[subject_index], label=subject if not plotted_label else None)
            ax.scatter(x, median, s=18, color=colors[subject_index], zorder=3)
            plotted_label = True

    ax.axhline(0.0, linestyle="--", linewidth=1)

    ax.set_xlabel(r"Standard GLM outer-CV $R^2$ $(\%)$")
    ax.set_ylabel(r"$\Delta R^2$ (percentage points)")
    ax.set_title(f"Improvement as a function of baseline accuracy: {METHOD_LABELS[method]}")
    ax.grid(alpha=0.5)
    ax.legend(title="Subject", loc="lower right" if method == "ica" else "upper right")

    _save_figure(fig, output_path)
    return fig

def plot_median_r2(median_r2_sub, output_path=None):
    subjects = list(median_r2_sub.keys())
    n_subjects = len(subjects)

    x = np.arange(n_subjects)
    group_width = 0.8
    bar_width = group_width / len(METHOD_ORDER)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    for method_index, method in enumerate(METHOD_ORDER):
        values = np.asarray([median_r2_sub[subject][method] * 100.0 for subject in subjects])
        
        positions = x - group_width / 2 + (method_index + 0.5) * bar_width
        bars = ax.bar(positions, values, width=bar_width * 0.9, label=METHOD_LABELS[method])

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_xlabel("Subject")
    ax.set_ylabel(r"Median outer-CV $R^2$ $(\%)$")
    ax.set_title("Median prediction accuracy across subjects")
    ax.axhline(0.0, linewidth=1)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

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

def plot_runtime(fit_runtime_sub, outer_cv_runtime_sub, output_path=None):
    subjects = list(fit_runtime_sub.keys())
    n_subjects = len(subjects)

    x = np.arange(n_subjects)
    group_width = 0.8
    bar_width = group_width / len(METHOD_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    for method_index, method in enumerate(METHOD_ORDER):
        positions = x - group_width / 2 + (method_index + 0.5) * bar_width

        fit_values = np.asarray([fit_runtime_sub[subject][method] for subject in subjects])
        outer_values = np.asarray([outer_cv_runtime_sub[subject][method] for subject in subjects])

        axes[0].bar(positions, fit_values, width=bar_width, label=METHOD_LABELS[method])
        axes[1].bar(positions, outer_values, width=bar_width, label=METHOD_LABELS[method])

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(subjects)
        ax.set_xlabel("Subject")
        ax.set_ylabel("Runtime (s)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_title("Full-data fit")
    axes[1].set_title("Outer-CV benchmark")
    axes[1].legend()

    fig.suptitle("Computational runtime")
    _save_figure(fig, output_path)
    return fig

def _common_abs_vmax(images, percentile=100.0):
    values = []

    for image in images.values():
        data = image.get_fdata()

        finite = np.abs(data[np.isfinite(data)])
        finite = finite[finite > 0] # Ignore zero-valued background voxels.
        if finite.size:
            values.append(finite)

    if not values:
        return 1.0
    values = np.concatenate(values)
    
    vmax = np.percentile(values, percentile)

    if not np.isfinite(vmax) or vmax <= 0:
        return 1.0
    return vmax


def _transparent_diverging_cmap(name="RdBu_r", n=256, alpha_power=1.5):
    """
    Diverging colormap whose opacity decreases toward zero.

    Large positive/negative values remain fully visible, while
    values close to zero allow the anatomical background to show.
    """
    positions = np.linspace(-1.0, 1.0, n)

    rgba = colormaps[name](np.linspace(0.0, 1.0, n))
    rgba[:, 3] = np.abs(positions) ** alpha_power

    return ListedColormap(rgba)

def _get_cut_coords(reference_img, mask_img, direction, n_cuts):
    return find_cut_slices(
        reference_img,
        direction=direction,
        n_cuts=n_cuts,
    )

def _plot_map_comparison(images, mask_img, background_img, title, threshold, output_path=None, display_mode="z", cut_coords=None, n_cuts=5, vmax_percentile=100.0, cmap="RdBu_r"):
    reference_img = images["standard_glm"]

    if cut_coords is None:
        cut_coords = _get_cut_coords(reference_img, mask_img, direction=display_mode, n_cuts=n_cuts)

    vmax = _common_abs_vmax(images, percentile=vmax_percentile)

    fig, axes = plt.subplots(len(METHOD_ORDER), 1, figsize=(13, 9), facecolor="black")

    for index, method in enumerate(METHOD_ORDER):
        ax = axes[index]
        ax.set_facecolor("black")

        plot_stat_map(
            stat_map_img=images[method],
            bg_img=background_img,
            display_mode=display_mode,
            cut_coords=cut_coords,
            threshold=threshold,
            vmax=vmax,
            symmetric_cbar=True,
            cmap=cmap,
            black_bg=True,
            dim=-0.5,
            title=None,
            annotate=True,
            draw_cross=False,
            colorbar=(index == len(METHOD_ORDER) - 1),
            axes=ax,
        )
        ax.set_title(METHOD_LABELS[method], color="white", fontsize=14, loc="left", pad=4)

    fig.suptitle(title, fontsize=18, color="white")
    fig.subplots_adjust(top=0.93, bottom=0.03, hspace=0.12)

    _save_figure(fig, output_path)
    return fig

def plot_beta_map_comparison(result, category, mask_img, t1_img, output_path=None, display_mode="z", cut_coords=None, n_cuts=5):
    images = {
        method: beta_to_img(
            task_coef=result.final_task_coef[method],
            category=category,
            mask_img=mask_img,
        )
        for method in METHOD_ORDER
    }

    beta_cmap = _transparent_diverging_cmap(name="RdBu_r", alpha_power=1.0)

    return _plot_map_comparison(
        images=images,
        mask_img=mask_img,
        background_img=t1_img,
        title=f"Beta maps: {category}",
        threshold=None,
        output_path=output_path,
        display_mode=display_mode,
        cut_coords=cut_coords,
        n_cuts=n_cuts,
        vmax_percentile=99.0,   # Avoid one extreme voxel destroying contrast in the entire map.
        cmap=beta_cmap,
    )

def plot_contrast_t_map_comparison(result, contrast, contrast_name, mask_img, t1_img, threshold=3.0, output_path=None, display_mode="z", cut_coords=None, n_cuts=5):
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
        vmax_percentile=100.0,
        cmap="RdBu_r",
    )

def plot_pca_selected_components(cv_scores_sub, best_num_sub, output_path=None):
    
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    
    for subject, r2_scores in cv_scores_sub.items():
        n_components = np.asarray(list(r2_scores.keys()))
        scores = np.asarray(list(r2_scores.values())) * 100.0

        best_n = best_num_sub[subject]
        best_score = r2_scores[best_n] * 100.0

        ax.plot(n_components, scores, linewidth=2, label=subject)
        ax.scatter(best_n, best_score, s=50)

    ax.set_xlabel("Number of PCA noise regressors")
    ax.set_ylabel(r"Cross-validated $R^2$ (%)")
    ax.set_title("Selection of PCA noise regressors")
    ax.set_xticks(n_components)
    ax.set_xticklabels(n_components)
    ax.grid(alpha=0.7)
    ax.legend()

    _save_figure(fig, output_path)
    return fig

def plot_ica_selected_components(q_by_run_sub, n_task_by_run_sub, n_nuisance_by_run_sub, output_path=None):
    subjects = list(q_by_run_sub.keys())
    n_subjects = len(subjects)

    group_centers = np.arange(n_subjects)
    group_width = 0.8

    # Dark = task, light = nuisance
    task_color = "#2F5D8A"
    nuisance_color = "#A9C5E8"

    fig, ax = plt.subplots(
        figsize=(10, 5),
        constrained_layout=True,
    )

    for subject_index, subject in enumerate(subjects):
        task_runs = np.asarray(n_task_by_run_sub[subject])
        nuisance_runs = np.asarray(n_nuisance_by_run_sub[subject])

        n_runs = len(task_runs)
        bar_width = group_width / n_runs
        run_positions = group_centers[subject_index] - group_width / 2 + (np.arange(n_runs) + 0.5) * bar_width

        ax.bar(run_positions, task_runs, width=bar_width * 0.95, color=task_color, edgecolor="black", linewidth=0.3)
        ax.bar(run_positions, nuisance_runs, width=bar_width * 0.95, bottom=task_runs, color=nuisance_color, edgecolor="black", linewidth=0.3)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(subjects)

    for i in range(n_subjects - 1):
        ax.axvline(i + 0.5, linestyle="--", linewidth=0.8, alpha=0.3)

    legend_handles = [
        Patch(facecolor=task_color, edgecolor="black", label="Task-related"),
        Patch(facecolor=nuisance_color, edgecolor="black", label="Nuisance"),
    ]
    ax.legend(handles=legend_handles, title="Component type", loc="lower right")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Number of ICA components")
    ax.set_title("ICA component classification across runs and subjects")
    ax.grid(axis="y", alpha=0.2)

    _save_figure(fig, output_path)
    return fig

def plot_ica_z_scores(z_scores_by_run_sub, threshold=0.0, output_path=None):
    rng = np.random.default_rng(42)
    subjects = list(z_scores_by_run_sub.keys())

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    for subject_index, subject in enumerate(subjects):
        z_scores = np.concatenate([np.asarray(z_run) for z_run in z_scores_by_run_sub[subject]])
        z_scores = z_scores[np.isfinite(z_scores)]

        group_width = 0.7

        n_runs = len(z_scores_by_run_sub[subject])
        run_offsets = np.linspace(-group_width / 2, group_width / 2, n_runs)

        for run_index, z_run in enumerate(z_scores_by_run_sub[subject]):
            z_run = np.asarray(z_run)
            z_run = z_run[np.isfinite(z_run)]

            local_jitter = rng.uniform(-0.015, 0.015, size=len(z_run))
            x = subject_index + run_offsets[run_index] + local_jitter

            ax.scatter(x, z_run, s=8, alpha=0.25)

    ax.axhline(threshold, linestyle="--", linewidth=1.2, label=rf"Classification threshold $c={threshold:g}$")
    ax.axhspan(ax.get_ylim()[0], threshold, alpha=0.2, label="Noise classification area")

    ax.set_xticks(np.arange(len(subjects)))
    ax.set_xticklabels(subjects)

    ax.set_xlabel("Subject")
    ax.set_ylabel(r"Task-association $z$-score")
    ax.set_title("ICA component task-association scores")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

    _save_figure(fig, output_path)
    
    return fig
