from pathlib import Path
import pandas as pd


METRICS_DIR = Path.cwd().parent / "contrast_metrics"


def calculate_contrast_metrics(model, stat_type, output_type, thresholds):
    """Calculates contrast metrics and saves to .csv file

    Args:
        model: GLMModel instance class (models/glm.py)
        stat_type (str): Can be 't', 'F', or 'all' (for both)
        output_type (str): Can be 'z_score', 'stat', 'p_value' or 'all'
        thresholds (Optional: dict[str]: int): dict with thresholds with output_type as key and float as value

    Returns:
        pd.DataFrame: DataFrame with calculated statistics
    """
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
