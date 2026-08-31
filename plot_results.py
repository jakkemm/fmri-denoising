import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from evaluation.maps import pairwise_contrast
from evaluation.pipeline import SubjectEvaluationResult
from evaluation.plots import (
    construct_cut_coords,
    plot_beta_map_comparison,
    plot_binned_r2_improvement,
    plot_contrast_t_map_comparison,
    plot_delta_r2_distribution,
    plot_ica_selected_components,
    plot_ica_z_scores,
    plot_median_r2,
    plot_pca_selected_components,
    plot_r2_scatter,
    plot_runtime,
    plot_snr_scatter,
)
from main import BASE_PATH
from utils.load_data import dataclass_from_pickle, load_data

RESULTS_PATH = Path.cwd() / "results"
SUBJECTS = [f"sub-{i}" for i in range(1, 7)]


def load_evaluation_result(subject):
    pickle_path = RESULTS_PATH / f"evaluation_{subject}.pkl"
    return dataclass_from_pickle(SubjectEvaluationResult, pickle_path)

def _close(fig):
    plt.close(fig)

def plot_subject(subject, beta_category="face", positive_contrast="face", negative_contrast="house", cut_coords=None):
    print(f"Plotting results for {subject}...")

    result = load_evaluation_result(subject)
    _, mask_img, t1_img = load_data(base_path=BASE_PATH, subject=subject)

    output_dir = RESULTS_PATH / "plots" / "individual"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Contrast t-map
    contrast = pairwise_contrast(
        positive_category=positive_contrast,
        negative_category=negative_contrast,
    )
    contrast_name = f"{positive_contrast} > {negative_contrast}"
    
    if cut_coords is None:
        cut_coords = construct_cut_coords(
            result=result, 
            contrast=contrast, 
            mask_img=mask_img, 
            display_mode="z"
        )

    # beta map
    fig = plot_beta_map_comparison(
        result=result,
        category=beta_category,
        mask_img=mask_img,
        t1_img=t1_img,
        cut_coords=cut_coords,
        subject_name=subject,
        output_path=output_dir / f"{subject}_beta_{beta_category}.png"
    )
    _close(fig)

    fig = plot_contrast_t_map_comparison(
        result=result,
        contrast=contrast,
        contrast_name=contrast_name,
        mask_img=mask_img,
        t1_img=t1_img,
        threshold=3.0,
        cut_coords=cut_coords,
        subject_name=subject,
        output_path=output_dir / f"{subject}_t_{positive_contrast}_vs_{negative_contrast}.png"
    )
    _close(fig)
    
    print(f"Plots saved to: {output_dir}")

def plot_groups_statistics():
    median_r2_sub = {}
    fit_runtime_sub = {}
    outer_cv_runtime_sub = {}
    candidate_mask_sub = {}
    r2_per_voxel_sub = {}
    delta_r2_vs_standard_sub = {}
    jackknife_snr_sub = {}
    
    for subject in SUBJECTS:
        result = load_evaluation_result(subject)
        
        median_r2_sub[subject] = result.median_r2
        fit_runtime_sub[subject] = result.fit_runtime_seconds
        outer_cv_runtime_sub[subject] = result.outer_cv_runtime_seconds
        candidate_mask_sub[subject] = result.candidate_mask
        r2_per_voxel_sub[subject] = result.r2_per_voxel
        delta_r2_vs_standard_sub[subject] = result.delta_r2_vs_standard
        jackknife_snr_sub[subject] = result.jackknife_snr
    
    output_dir = RESULTS_PATH / "plots" / "group"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # MEDIAN R2
    fig = plot_median_r2(
        median_r2_sub=median_r2_sub,
        output_path=output_dir / "median_r2.png"
    )
    _close(fig)
    
    # RUNTIME
    fig = plot_runtime(
        fit_runtime_sub=fit_runtime_sub,
        outer_cv_runtime_sub=outer_cv_runtime_sub,
        output_path=output_dir / "runtime.png"
    )
    _close(fig)

    # R2 IMPROVEMENT
    fig = plot_binned_r2_improvement(
        r2_per_voxel_sub=r2_per_voxel_sub,
        candidate_mask_sub=candidate_mask_sub,
        method="glm_pca",
        bin_width_pp=10.0,
        output_path=output_dir / "r2_improvement_binned_glm_pca.png"
    )
    _close(fig)    
    fig = plot_binned_r2_improvement(
        r2_per_voxel_sub=r2_per_voxel_sub,
        candidate_mask_sub=candidate_mask_sub,
        method="ica",
        bin_width_pp=10.0,
        output_path=output_dir / "r2_improvement_binned_ica.png"
    )
    _close(fig)

    # R2 DISTRIBUTION
    fig = plot_delta_r2_distribution(
        delta_r2_vs_standard_sub=delta_r2_vs_standard_sub,
        candidate_mask_sub=candidate_mask_sub,
        method="glm_pca",
        output_path=output_dir / "delta_r2_distribution_glm_pca.png"
    )
    _close(fig)
    fig = plot_delta_r2_distribution(
        delta_r2_vs_standard_sub=delta_r2_vs_standard_sub,
        candidate_mask_sub=candidate_mask_sub,
        method="ica",
        output_path=output_dir / "delta_r2_distribution_ica.png"
    )
    _close(fig)
    
    # R2 SCATTER
    fig = plot_r2_scatter(
        r2_per_voxel_sub=r2_per_voxel_sub,
        candidate_mask_sub=candidate_mask_sub,
        method="glm_pca",
        output_path=output_dir / "r2_scatter_glm_pca.png"
    )
    _close(fig)
    fig = plot_r2_scatter(
        r2_per_voxel_sub=r2_per_voxel_sub,
        candidate_mask_sub=candidate_mask_sub,
        method="ica",
        output_path=output_dir / "r2_scatter_ica.png"
    )
    _close(fig)


    # JACKKNIFE SNR
    fig = plot_snr_scatter(
        snr_by_method_sub=jackknife_snr_sub,
        candidate_mask_sub=candidate_mask_sub,
        method="glm_pca",
        output_path=output_dir / "snr_scatter_glm_pca.png"
    )
    _close(fig)
    fig = plot_snr_scatter(
        snr_by_method_sub=jackknife_snr_sub,
        candidate_mask_sub=candidate_mask_sub,
        method="ica",
        output_path=output_dir / "snr_scatter_ica.png"
    )
    _close(fig)

def plot_component_comparison():
    pca_cv_scores = {}
    pca_best_component = {}
    ica_model_order = {}
    ica_n_task = {}
    ica_n_nuis = {}
    ica_z_scores = {}
    
    for subject in SUBJECTS:
        result = load_evaluation_result(subject)
        
        pca_cv_scores[subject] = result.final_method_specific_data["glm_pca"]["cv_scores"]
        pca_best_component[subject] = result.final_method_specific_data["glm_pca"]["selected_n_components"]
        ica_model_order[subject] = result.final_method_specific_data["ica"]["q_by_run"]
        ica_n_task[subject] = result.final_method_specific_data["ica"]["n_task_by_run"]
        ica_n_nuis[subject] = result.final_method_specific_data["ica"]["n_nuisance_by_run"]
        ica_z_scores[subject] = result.final_method_specific_data["ica"]["z_scores_by_run"]
        
    output_dir = RESULTS_PATH / "plots" / "group"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plot_pca_selected_components(
        cv_scores_sub=pca_cv_scores,
        best_num_sub=pca_best_component,
        output_path=output_dir / "pca_component_comparison.png"
    )
    _close(fig)
    
    fig = plot_ica_selected_components(
        q_by_run_sub=ica_model_order,
        n_task_by_run_sub=ica_n_task,
        n_nuisance_by_run_sub=ica_n_nuis,
        output_path=output_dir / "ica_component_comparison.png"
    )
    _close(fig)
    
    fig = plot_ica_z_scores(
        z_scores_by_run_sub=ica_z_scores,
        threshold=0.0,
        output_path=output_dir / "ica_z_scores_comparison.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--beta_category", type=str, default="face")
    parser.add_argument("--contrast", nargs=2, default=["face", "house"], metavar=("POSITIVE", "NEGATIVE"))
    parser.add_argument("--cuts", nargs="+", type=float, default=None)
    
    parser.add_argument("--individual", action="store_true")
    parser.add_argument("--group", action="store_true")
    parser.add_argument("--components", action="store_true")
    parser.add_argument("--subject", type=str, default=None)

    args = parser.parse_args()

    if args.individual:
        for subject in SUBJECTS:
            plot_subject(
                subject=subject,
                beta_category=args.beta_category,
                positive_contrast=args.contrast[0],
                negative_contrast=args.contrast[1],
                cut_coords=args.cuts
            )
    if args.subject is not None:
        plot_subject(
            subject=args.subject,
            beta_category=args.beta_category,
            positive_contrast=args.contrast[0],
            negative_contrast=args.contrast[1],
            cut_coords=args.cuts
        )

    if args.group:
        plot_groups_statistics()
    if args.components:
        plot_component_comparison()
