"""
activation_maps_glm.py

Create t-value activation maps from Nilearn first-level GLM.

Produces:
    - Standard GLM t-map
    - Denoised-ish GLM t-map with PCA-like high-variance confounds
    - Side-by-side activation map figure, thresholded at t > 3

Requires:
    pip install nilearn nibabel pandas numpy matplotlib

Assumes:
    load_data.py is in the same directory or available on PYTHONPATH.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.image import (
    concat_imgs,
    mean_img,
    math_img,
    resample_to_img,
    high_variance_confounds,
)
from nilearn.plotting import plot_stat_map, plot_design_matrix
from nilearn.reporting import get_clusters_table

from utils.load_data import load_data


# ============================================================
# CONFIG
# ============================================================

BASE_PATH = "/Users/jakubkempa/Documents/magisterka/data/ds000105_R2.0.2"
SUBJECT = "sub-1"


# "mean_func" is safest.
# "t1" looks better only if T1 and fMRI are aligned in the same space.
BACKGROUND = "t1"   # options: "mean_func", "t1"

T_THRESHOLD = 3.0

# Denoised-ish model. This is not exact GLMdenoise, but a cheap PCA-confound approximation.
N_CONFOUNDS_DENOISED = 5

OUTPUT_DIR = Path("results_activation_maps") / SUBJECT

# Plotting options
DISPLAY_MODE = "z"
CUT_COORDS = 6
BLACK_BG = True


# ============================================================
# UTILITIES
# ============================================================

def safe_name(text):
    """Make condition names usable as filenames, because filesystems are fragile little beasts."""
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)
    return text.strip("_")


def matlab_round_positive(x):
    """
    MATLAB-style rounding for positive numbers.
    MATLAB round(2.5) -> 3, Python round(2.5) -> 2.
    GLMdenoise used MATLAB, because apparently one rounding convention was not enough.
    """
    return int(np.floor(x + 0.5))


def prepare_events(events_df):
    """
    Convert a BIDS-like events.tsv DataFrame into the format expected by Nilearn:
        onset, duration, trial_type, optional modulation.
    """
    events = events_df.copy()

    if "onset" not in events.columns:
        raise ValueError("events.tsv must contain an 'onset' column.")

    if "duration" not in events.columns:
        events["duration"] = 0.0

    if "trial_type" not in events.columns:
        candidates = ["condition", "stim_type", "event_type", "task", "trial"]
        found = next((col for col in candidates if col in events.columns), None)

        if found is None:
            events["trial_type"] = "event"
        else:
            events["trial_type"] = events[found].astype(str)

    keep_cols = ["onset", "duration", "trial_type"]

    if "modulation" in events.columns:
        keep_cols.append("modulation")

    events = events[keep_cols].copy()
    events["onset"] = pd.to_numeric(events["onset"])
    events["duration"] = pd.to_numeric(events["duration"]).fillna(0.0)
    events["trial_type"] = events["trial_type"].astype(str)

    return events.sort_values("onset").reset_index(drop=True)


def make_design_matrices(
    images,
    events_list,
    tr,
    hrf_model="spm",
    use_high_variance_confounds=False,
    n_confounds=0,
    drift_model=None
):
    """
    Create one design matrix per run.

    GLMdenoise-inspired structure:
        task regressors convolved with HRF
        + optional polynomial drift regressors
        + optional PCA-like nuisance regressors
    """
    if len(images) != len(events_list):
        raise ValueError(
            f"Number of images ({len(images)}) and events files ({len(events_list)}) differs."
        )

    design_matrices = []

    for run_idx, (img, events_df) in enumerate(zip(images, events_list), start=1):
        n_scans = img.shape[-1]
        frame_times = np.arange(n_scans) * tr

        run_duration_min = (n_scans * tr) / 60.0
        drift_order = matlab_round_positive(run_duration_min / 2.0)

        events = prepare_events(events_df)

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
            events=events,
            hrf_model=hrf_model,
            drift_model=drift_model,
            drift_order=drift_order,
            add_regs=add_regs,
            add_reg_names=add_reg_names,
            min_onset=0,
        )

        print(
            f"Run {run_idx:02d}: "
            f"n_scans={n_scans}, "
            f"duration={run_duration_min:.2f} min, "
            f"drift_order={drift_order}, "
            f"design shape={design_matrix.shape}"
        )

        design_matrices.append(design_matrix)

    return design_matrices


def fit_first_level_model(images, design_matrices, tr, subject_label):
    """Fit Nilearn first-level GLM."""
    model = FirstLevelModel(
        noise_model="ar1",
        standardize=False,
        signal_scaling=0,
        minimize_memory=False,
        subject_label=subject_label,
    )

    model = model.fit(
        run_imgs=images,
        design_matrices=design_matrices,
    )

    return model


def get_task_regressors(design_matrix):
    """
    Heuristic to identify likely task regressors.
    Removes nuisance/drift/confound columns.
    """
    nuisance_prefixes = (
        "constant",
        "drift",
        "poly",
        "hvc_",
    )

    nuisance_exact = {
        "intercept",
    }

    task_cols = []

    for col in design_matrix.columns:
        col_lower = col.lower()

        if col_lower in nuisance_exact:
            continue

        if any(col_lower.startswith(prefix) for prefix in nuisance_prefixes):
            continue

        task_cols.append(col)

    return task_cols


def choose_condition_if_needed(design_matrices, condition_name):
    """Use chosen condition or fall back to first likely task regressor."""
    all_task_cols = []

    for dm in design_matrices:
        all_task_cols.extend(get_task_regressors(dm))

    unique_task_cols = sorted(set(all_task_cols))

    print("\nAvailable likely task regressors:")
    for col in unique_task_cols:
        print(f"  - {col}")

    if condition_name is None:
        if not unique_task_cols:
            raise ValueError(
                "Could not automatically identify task regressors. "
                "Inspect design_matrices[0].columns and set CONDITION_NAME manually."
            )

        chosen = unique_task_cols[0]
        print(f"\nCONDITION_NAME was None, using first task regressor: {chosen}")
        return chosen

    if condition_name not in unique_task_cols:
        raise ValueError(
            f"Condition '{condition_name}' not found in likely task regressors. "
            "Check printed list above and set CONDITION_NAME."
        )

    return condition_name


def make_condition_contrast(design_matrices, condition_name):
    """
    Create one contrast vector per run:
        condition_name > baseline
    """
    contrasts = []
    found_anywhere = False

    for dm in design_matrices:
        contrast = np.zeros(dm.shape[1])

        if condition_name in dm.columns:
            contrast[dm.columns.get_loc(condition_name)] = 1.0
            found_anywhere = True

        contrasts.append(contrast)

    if not found_anywhere:
        raise ValueError(
            f"Condition '{condition_name}' was not found in any design matrix."
        )

    return contrasts


def compute_t_map(model, condition_name):
    """Compute t-statistic map for one condition."""
    contrast = make_condition_contrast(
        model.design_matrices_,
        condition_name=condition_name,
    )

    t_map = model.compute_contrast(
        contrast,
        stat_type="t",
        output_type="stat",
    )

    return t_map


def positive_threshold_t_map(t_map, threshold):
    """Keep only positive t-values above threshold."""
    return math_img(
        f"img * (img > {threshold})",
        img=t_map,
    )


def make_background(images, t1_img, background_mode):
    """
    Create background image.

    mean_func:
        safest, because activation maps are in functional space.

    t1:
        prettier, but only valid if T1 and fMRI are aligned.
    """
    if background_mode == "mean_func":
        bg_img = mean_img(concat_imgs(images))
        print("\nUsing mean functional image as background.")
        return bg_img

    if background_mode == "t1":
        print(
            "\nUsing T1 image as background. "
            "If the overlay looks shifted, the images are not coregistered."
        )
        return t1_img

    raise ValueError("BACKGROUND must be either 'mean_func' or 't1'.")


def maybe_resample_to_background(stat_img, bg_img, background_mode):
    """
    If using T1 background, resample stat image to T1 grid.

    This is resampling, not registration.
    If the images are not already aligned in world coordinates, this will not fix it.
    """
    if background_mode == "t1":
        return resample_to_img(
            stat_img,
            bg_img,
            interpolation="continuous",
            force_resample=True,
            copy_header=True,
        )

    return stat_img


def plot_comparison(
    t_standard_thr,
    t_denoised_thr,
    bg_img,
    condition_name,
    output_file,
):
    """Plot Standard GLM and denoised-ish GLM side by side."""
    fig = plt.figure(figsize=(15, 5))

    plot_stat_map(
        t_standard_thr,
        bg_img=bg_img,
        threshold=1e-6,
        display_mode=DISPLAY_MODE,
        cut_coords=CUT_COORDS,
        colorbar=True,
        black_bg=BLACK_BG,
        title=f"Standard GLM: {condition_name}, t > {T_THRESHOLD}",
        figure=fig,
        axes=(0.00, 0.00, 0.48, 1.00),
    )

    plot_stat_map(
        t_denoised_thr,
        bg_img=bg_img,
        threshold=1e-6,
        display_mode=DISPLAY_MODE,
        cut_coords=CUT_COORDS,
        colorbar=True,
        black_bg=BLACK_BG,
        title=f"Denoised-ish GLM: {condition_name}, t > {T_THRESHOLD}",
        figure=fig,
        axes=(0.52, 0.00, 0.48, 1.00),
    )

    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_file}")


def save_design_matrix_plots(design_standard, design_denoised, output_dir):
    """Save first-run design matrix plots for sanity checking."""
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_design_matrix(design_standard[0], axes=ax)

    fig.savefig(
        output_dir / "design_matrix_standard_run01.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_design_matrix(design_denoised[0], axes=ax)

    fig.savefig(
        output_dir / "design_matrix_denoised_run01.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)

def save_cluster_table(t_map, output_file):
    """Save simple cluster summary for positive t-values."""
    table = get_clusters_table(
        t_map,
        stat_threshold=T_THRESHOLD,
        cluster_threshold=10,
        two_sided=False,
    )

    table.to_csv(output_file, index=False)
    print(f"Saved cluster table: {output_file}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for {SUBJECT}...")
    images, events, t1_img, tr = load_data(BASE_PATH, subject=SUBJECT)

    print(f"Loaded {len(images)} runs.")
    print(f"TR = {tr}")

    print("\nCreating Standard GLM design matrices...")
    design_standard = make_design_matrices(
        images=images,
        events_list=events,
        tr=tr,
        hrf_model="spm",
        use_high_variance_confounds=False,
        n_confounds=0,
        drift_model=None
    )

    condition_name = choose_condition_if_needed(
        design_standard,
        CONDITION_NAME,
    )

    print("\nCreating denoised-ish GLM design matrices...")
    design_denoised = make_design_matrices(
        images=images,
        events_list=events,
        tr=tr,
        hrf_model="spm",
        use_high_variance_confounds=True,
        n_confounds=N_CONFOUNDS_DENOISED,
        drift_model="polynomial"
    )

    print("\nSaving design matrix plots...")
    save_design_matrix_plots(
        design_standard,
        design_denoised,
        OUTPUT_DIR,
    )

    print("\nFitting Standard GLM...")
    model_standard = fit_first_level_model(
        images=images,
        design_matrices=design_standard,
        tr=tr,
        subject_label=f"{SUBJECT}_standard",
    )

    print("\nFitting denoised-ish GLM...")
    model_denoised = fit_first_level_model(
        images=images,
        design_matrices=design_denoised,
        tr=tr,
        subject_label=f"{SUBJECT}_denoised_hvc{N_CONFOUNDS_DENOISED}",
    )

    print(f"\nComputing t-map for condition: {condition_name}")
    t_standard = compute_t_map(
        model_standard,
        condition_name=condition_name,
    )

    t_denoised = compute_t_map(
        model_denoised,
        condition_name=condition_name,
    )

    condition_safe = safe_name(condition_name)

    t_standard_file = OUTPUT_DIR / f"tmap_standard_{condition_safe}.nii.gz"
    t_denoised_file = OUTPUT_DIR / f"tmap_denoised_hvc{N_CONFOUNDS_DENOISED}_{condition_safe}.nii.gz"

    t_standard.to_filename(t_standard_file)
    t_denoised.to_filename(t_denoised_file)

    print(f"Saved t-map: {t_standard_file}")
    print(f"Saved t-map: {t_denoised_file}")

    print("\nCreating background image...")
    bg_img = make_background(
        images=images,
        t1_img=t1_img,
        background_mode=BACKGROUND,
    )

    print("\nPreparing thresholded maps...")
    t_standard_for_plot = maybe_resample_to_background(
        t_standard,
        bg_img,
        BACKGROUND,
    )

    t_denoised_for_plot = maybe_resample_to_background(
        t_denoised,
        bg_img,
        BACKGROUND,
    )

    t_standard_thr = positive_threshold_t_map(
        t_standard_for_plot,
        T_THRESHOLD,
    )

    t_denoised_thr = positive_threshold_t_map(
        t_denoised_for_plot,
        T_THRESHOLD,
    )

    comparison_png = OUTPUT_DIR / (
        f"activation_comparison_{condition_safe}_"
        f"t_gt_{str(T_THRESHOLD).replace('.', '_')}_bg_{BACKGROUND}.png"
    )

    print("\nPlotting activation maps...")
    plot_comparison(
        t_standard_thr=t_standard_thr,
        t_denoised_thr=t_denoised_thr,
        bg_img=bg_img,
        condition_name=condition_name,
        output_file=comparison_png,
    )

    print("\nSaving cluster tables...")
    save_cluster_table(
        t_standard_for_plot,
        OUTPUT_DIR / f"clusters_standard_{condition_safe}.csv",
    )

    save_cluster_table(
        t_denoised_for_plot,
        OUTPUT_DIR / f"clusters_denoised_hvc{N_CONFOUNDS_DENOISED}_{condition_safe}.csv",
    )

    print("\nDone. The blobs have been generated. May they be aligned and statistically less cursed.")


if __name__ == "__main__":
    main()