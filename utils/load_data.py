import nibabel as nib
import pandas as pd
import numpy as np

from pathlib import Path
import json

def load_data(base_path, subject):
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
    t1_file = list(t1_path.iterdir())[0]
    
    # Repetition Time
    tr_file = list(base_path.glob("*bold.json"))[0]
    with open(tr_file, "r") as f:
        metadata = json.load(f)
    
    images = [nib.load(str(f)) for f in fmri_files]
    events = [pd.read_csv(f, sep="\t") for f in events_files]
    t1_img = nib.load(t1_file)
    tr = float(metadata["RepetitionTime"])
    return images, events, t1_img, tr

def img_to_2d(img):
    data = img.get_fdata(dtype=np.float32)
    n_scans = data.shape[3]
    Y = data.reshape(-1, n_scans).T          # shape: (time, voxels)
    mean_signal = Y.mean(axis=0)             # shape: (voxels,)
    return Y, mean_signal, n_scans


if __name__ == "__main__":
    base_path = "/Users/jakubkempa/Documents/magisterka/ds000105_R2.0.2"

    all_data = load_data(base_path, subject="sub-1")
