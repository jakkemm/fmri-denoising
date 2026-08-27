import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from evaluation.maps import pairwise_contrast
from evaluation.metrics import summarize_normalized_performance
from evaluation.pipeline import SubjectEvaluationResult
from evaluation.plots import (
    plot_beta_map_comparison,
    plot_binned_r2_improvement,
    plot_contrast_t_map_comparison,
    plot_delta_r2_distribution,
    plot_median_r2,
    plot_normalized_performance,
    plot_r2_scatter,
    plot_runtime,
    plot_snr_scatter,
)
from main import BASE_PATH
from utils.load_data import dataclass_from_pickle, load_data

RESULTS_PATH = Path.cwd() / "results"


def load_evaluation_result(subject):
    pickle_path = RESULTS_PATH / f"evaluation_{subject}.pkl"
    return dataclass_from_pickle(SubjectEvaluationResult, pickle_path)

def _close(fig):
    plt.close(fig)

def plot_subject(subject, beta_category="face", positive_contrast="face", negative_contrast="scrambledpix", cut_coords=None):
    print(f"Plotting results for {subject}...")

    result = load_evaluation_result(subject)
    _, mask_img, t1_img = load_data(base_path=BASE_PATH, subject=subject)

    output_dir = RESULTS_PATH / "plots" / subject
    output_dir.mkdir(parents=True, exist_ok=True)

    # Outer-CV R2 scatter
    fig = plot_r2_scatter(
        r2_per_voxel=result.r2_per_voxel,
        candidate_mask=result.candidate_mask,
        output_path=output_dir / "r2_scatter.png"
    )
    _close(fig)

    # Delta R2 distribution
    fig = plot_delta_r2_distribution(
        delta_r2_vs_standard=result.delta_r2_vs_standard,
        candidate_mask=result.candidate_mask,
        output_path=output_dir / "delta_r2_distribution.png"
    )
    _close(fig)

    # Binned R2 improvement
    fig = plot_binned_r2_improvement(
        r2_per_voxel=result.r2_per_voxel,
        candidate_mask=result.candidate_mask,
        bin_width_pp=10.0,  # TODO: consider changing this to 5.0 if R2 range is narrower
        output_path=output_dir / "r2_improvement_binned.png"
    )
    _close(fig)
    
    # Median R2
    fig = plot_median_r2(
        median_r2=result.median_r2,
        output_path=output_dir / "median_r2.png"
    )
    _close(fig)

    # Jackknife SNR
    fig = plot_snr_scatter(
        snr_by_method=result.jackknife_snr,
        candidate_mask=result.candidate_mask,
        output_path=output_dir / "snr_scatter.png"
    )
    _close(fig)

    # Runtime
    fig = plot_runtime(
        fit_runtime_seconds=result.fit_runtime_seconds,
        outer_cv_runtime_seconds=result.outer_cv_runtime_seconds,
        output_path=output_dir / "runtime.png"
    )
    _close(fig)

    # beta map
    fig = plot_beta_map_comparison(
        result=result,
        category=beta_category,
        mask_img=mask_img,
        t1_img=t1_img,
        cut_coords=cut_coords,
        output_path=output_dir / f"beta_{beta_category}.png"
    )
    _close(fig)

    # Contrast t-map
    contrast = pairwise_contrast(
        positive_category=positive_contrast,
        negative_category=negative_contrast,
    )
    contrast_name = f"{positive_contrast} > {negative_contrast}"

    fig = plot_contrast_t_map_comparison(
        result=result,
        contrast=contrast,
        contrast_name=contrast_name,
        mask_img=mask_img,
        t1_img=t1_img,
        # Same threshold as Figure 5
        # in GLMdenoise.
        threshold=3.0,  # Same threshold as Figure 5 in GLMdenoise
        cut_coords=cut_coords,
        output_path=output_dir / f"t_{positive_contrast}_vs_{negative_contrast}.png"
    )
    _close(fig)
    
    print(f"Plots saved to: {output_dir}")


def plot_group(subjects):
    normalized_by_subject = []

    for subject in subjects:
        result = load_evaluation_result(subject)
        normalized_by_subject.append(result.normalized_performance)

        mean_performance, sem_performance = summarize_normalized_performance(normalized_by_subject)

    output_dir = RESULTS_PATH / "plots" / "group"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_normalized_performance(
        mean_performance=mean_performance,
        sem_performance=sem_performance,
        output_path=output_dir / "normalized_performance.png"
    )
    _close(fig)

    print("Mean normalized performance:")

    for method, value in mean_performance.items():
        print(f"  {method}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--subject", type=str)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--beta_category", type=str, default="face")
    parser.add_argument("--contrast", nargs=2, default=["face", "scrambledpix"], metavar=("POSITIVE", "NEGATIVE"))
    parser.add_argument("--cuts", nargs="+", type=float, default=None)

    args = parser.parse_args()

    if args.subject is not None:
        plot_subject(
            subject=args.subject,
            beta_category=args.beta_category,
            positive_contrast=args.contrast[0],
            negative_contrast=args.contrast[1],
            cut_coords=args.cuts
        )

    if args.subjects is not None:
        plot_group(subjects=args.subjects)
