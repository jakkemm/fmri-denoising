import dataclasses
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.masking import apply_mask, compute_multi_epi_mask

from utils.constants import RawData


def _load_raw_imgs(dir_path):
    """Load raw BOLD signals"""
    
    fmri_path = dir_path / "func"
    fmri_files = sorted(fmri_path.glob("*_bold.nii.gz"))
    raw_images = [nib.load(str(y)) for y in fmri_files]
    return raw_images

def _load_events_dfs(dir_path):
    """Load events.tsv files as DataFrames"""
    
    fmri_path = dir_path / "func"
    events_files = sorted(fmri_path.glob("*_events.tsv"))
    events_dfs = [pd.read_csv(ev, sep="\t") for ev in events_files]
    return events_dfs

def _load_t1_img(dir_path):
    t1_path = dir_path / "anat"
    t1_file = next(t1_path.glob("*_T1w.nii.gz"))
    t1_img = nib.load(t1_file)
    return t1_img

def _load_mask_imgs(dir_path):
    mask_file = dir_path / "mask.nii.gz"
    
    if not mask_file.is_file():
        _create_masks(dir_path)
    
    mask_img = nib.load(mask_file)
    return mask_img

def _create_masks(dir_path):
    raw_imgs = _load_raw_imgs(dir_path)
    mask_img = compute_multi_epi_mask(raw_imgs)
    
    mask_path = dir_path / "mask.nii.gz"
    nib.save(mask_img, mask_path)


def load_data(base_path, subject):
    """
    Function for loading fMRI data. 
    Requires absolute path to BIDS directory and name of directory for given subject, 
    i.e. `load_data("~/Data/ds000105", "sub-1")`

    Returns:
        - list of RunData dataclasses containing 2D masked fMRI data and events_df
        - raw mask image
        - T1 high-resolution image
    """
    base_path = Path(base_path)
    
    if not base_path.is_absolute():
        raise ValueError(f"{base_path} is not an absolute path.")
    
    dir_path = base_path / subject
    if not dir_path.exists():
        raise FileNotFoundError(f"{dir_path} does not exist")

    raw_images = _load_raw_imgs(dir_path)
    events_dfs = _load_events_dfs(dir_path)
    mask_img = _load_mask_imgs(dir_path)
    t1_img = _load_t1_img(dir_path)
    
    runs = []
    for img, ev in zip(raw_images, events_dfs):
        Y = apply_mask(img, mask_img)
        
        run = RawData(
            Y=Y.astype(np.float32),
            events_df=ev
        )
        runs.append(run)
    
    return runs, mask_img, t1_img

def img_to_2d(img):
    data = img.get_fdata(dtype=np.float32)
    n_scans = data.shape[3]
    Y = data.reshape(-1, n_scans).T          # shape: (time, voxels)
    return Y

def dataclass_to_pickle(dataclass_obj, pickle_path):
    with open(pickle_path, "wb") as f:
        pickle.dump(dataclasses.asdict(dataclass_obj), f)

def dataclass_from_pickle(dataclass_type, pickle_path):
    with open(pickle_path, "rb") as f:
        return dataclass_type(**pickle.load(f))


if __name__ == "__main__":
    base_path = "/Users/jakubkempa/Documents/magisterka/ds000105_R2.0.2"

    all_data = load_data(base_path, subject="sub-1")
