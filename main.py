from pathlib import Path
import pandas as pd

from utils.load_data import load_data
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

def calculate_metrics(model, stat_type, output_type, thresholds):
    valid_stat_types = ("t", "F", "all")
    valid_output_types = ("z_score", "stat", "p_value", "all")
    
    stat_types = valid_stat_types[:-1] if stat_type == "all" else [stat_type]
    output_types = valid_output_types[:-1] if output_type == "all" else [output_type]
    
    all_metrics = []
    
    for stat_type_ in stat_types:
        for output_type_ in output_types:
            threshold = thresholds.get(output_type_)
            
            df = model.compute_contrast_metrics_for_condition(
                stat_type=stat_type_, output_type=output_type_, threshold=threshold
            )
            all_metrics.append(df)
    
    full_df = pd.concat(all_metrics, ignore_index=True)
    
    metrics_fname = f"{model.model_name}_{model.label}_metrics.csv"
    full_df.to_csv(METRICS_DIR / metrics_fname, index=None)
    
    return full_df
    

if __name__ == "__main__":
    fmri_data_dir = DATA_DIR / "ds000105_R2.0.2"
    subject = "sub-1"
    images, events, t1_img, tr = load_data(fmri_data_dir, subject=subject)

    glm_standard = GLMModel(
        images, events, t1_img, tr,
        label="sub-01", 
        model_name="standard",
        verbose=True
    ).make_design_matrices(
        hrf_model="spm",
        use_high_variance_confounds=False,
        n_confounds=0,
        drift_model=None
    ).fit(noise_model="ar1")
    
    calculate_metrics(
        model=glm_standard,
        stat_type="all",
        output_type="all",
        thresholds=SAMPLE_THRESHOLDS
    )
