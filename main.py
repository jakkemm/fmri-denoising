from pathlib import Path

from utils.load_data import load_data
from utils.models import GLMdenoiser

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"

if __name__ == "__main__":
    fmri_data_dir = DATA_DIR / "ds000105_R2.0.2"
    subject = "sub-1"
    all_data = load_data(fmri_data_dir, subject=subject)

    runs_to_analyze = list(range(3))

    glm = GLMdenoiser(*all_data)
    result = glm.full_workflow(max_noise=10, n_boot=100, runs_num=len(runs_to_analyze))
