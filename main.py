"""
This is the main module, in which all of the necessary calculations are perfomed.

The analyzed models will be expanded and analyzed for the thesis.
Current list of models is as follows:

    1. GLM Standard - design matrix only has task regressors.
    2. GLM Denoise - design matrix has task + high variance confounds + drift regressors.
"""

from pathlib import Path

from utils.load_data import load_data
from utils import metrics
from utils import visualize
from models.glm import GLMModel

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
METRICS_DIR = BASE_DIR / "contrast_metrics"

SAMPLE_THRESHOLDS = {
    "z_score": 3.0,
    "stat": 3.0,
    "p_value": 0.05,
}

def run_sample_glm_models(images, events, t1_img, tr):
    """Runs calculations for analyzed models."""
    
    glm_standard = (
        GLMModel(images, events, t1_img, tr, label="sub-01", model_name="standard", verbose=True)
        .make_design_matrices(hrf_model="spm", use_high_variance_confounds=False, n_confounds=0, drift_model=None)
        .fit(noise_model="ar1")
    )
    
    glm_denoise = (
        GLMModel(images, events, t1_img, tr, label="sub-01",  model_name="GLMdenoise", verbose=True)
        .make_design_matrices(hrf_model="spm", use_high_variance_confounds=True, n_confounds=5, drift_model="polynomial")
        .fit(noise_model="ar1")
    )
    
    for m in [glm_standard, glm_denoise]:
        metrics.calculate_contrast_metrics(
            model=m,
            stat_type="all",
            output_type="all",
            thresholds=SAMPLE_THRESHOLDS
        )

def run_denoising_glm(images, events, t1_img, tr):
    """Runs denoising for analyzed models."""
    
    standard_denoised = (
        GLMModel(images, events, t1_img, tr, label="sub-01", model_name="standard", verbose=True)
        .make_design_matrices(hrf_model="spm", use_high_variance_confounds=False, n_confounds=0, drift_model=None)
        .get_denoised_images()
    )
    hvc_drift_denoised = (
        GLMModel(images, events, t1_img, tr, label="sub-01",  model_name="GLMdenoise", verbose=True)
        .make_design_matrices(hrf_model="spm", use_high_variance_confounds=True, n_confounds=5, drift_model="polynomial")
        .get_denoised_images()
    )
    
    denoised = [standard_denoised, hvc_drift_denoised]
    visualize.plot_tsnr(denoised)
    


if __name__ == "__main__":
    fmri_data_dir = DATA_DIR / "ds000105_R2.0.2"
    subject = "sub-1"
    images, events, t1_img, tr = load_data(fmri_data_dir, subject=subject)
    breakpoint()
    
    run_denoising_glm(images, events, t1_img, tr)
    # run_sample_glm_models(images, events, t1_img, tr)
