from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from general_linear_model.constants import RawData


def load_data(base_path, subject):
    """
    Function for loading fMRI data. 
    Requires absolute path to BIDS directory and name of directory for given subject, 
    i.e. `load_data("~/Data/ds000105", "sub-1")`

    Returns:
        - list of RunData dataclasses containing 2D fMRI data and events_df
        - T1 high-resolution image
    """
    base_path = Path(base_path)
    
    if not base_path.is_absolute():
        raise ValueError(f"{base_path} is not an absolute path.")
    
    dir_path = base_path / subject
    if not dir_path.exists():
        raise FileNotFoundError(f"{dir_path} does not exist")

    # BOLD signals with events file
    fmri_path = dir_path / "func"
    fmri_files = sorted(fmri_path.glob("*.nii.gz"))
    events_files = sorted(fmri_path.glob("*.tsv"))
    
    # T1 images
    t1_path = dir_path / "anat"
    t1_file = next(t1_path.iterdir())
    t1_img = nib.load(t1_file)
    
    runs = []
    
    for y, ev in zip(fmri_files, events_files):
        image = nib.load(str(y))
        run = RawData(
            Y=img_to_2d(image),
            events_df=pd.read_csv(ev, sep="\t")
        )
        
        runs.append(run)
    
    return runs, t1_img

def img_to_2d(img):
    data = img.get_fdata(dtype=np.float32)
    n_scans = data.shape[3]
    Y = data.reshape(-1, n_scans).T          # shape: (time, voxels)
    return Y


if __name__ == "__main__":
    base_path = "/Users/jakubkempa/Documents/magisterka/ds000105_R2.0.2"

    all_data = load_data(base_path, subject="sub-1")
