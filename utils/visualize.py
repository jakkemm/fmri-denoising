import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib
from nilearn import plotting

from pathlib import Path

PLOTS_DIR = Path.cwd() / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def _compute_tsnr_map(img, eps=1e-8):
    data = img.get_fdata(dtype=np.float32)
    
    mean_signal = np.mean(data, axis=-1)
    std_dev = np.std(data, axis=-1, ddof=1)
    
    tsnr_map = np.divide(
        mean_signal, 
        std_dev,
        out=np.zeros_like(mean_signal, dtype=np.float32),
        where=std_dev > eps,
    )
    tsnr_map = np.nan_to_num(tsnr_map, nan=0.0, posinf=0.0, neginf=0.0)
    
    return tsnr_map.astype(np.float32)


def plot_tsnr(images, run_idx=0, t1_img=None):
    """
    Function for creating tSNR plots.
    This can be a comparison, if the provided argument `images` is a list of lists.
    """
    
    if not all([isinstance(img, list) for img in images]):
        images = [images]
    
    z_planes = []
        
    fig, ax = plt.subplots(nrows=len(images), ncols=3, figsize=(12, 5*len(images)))
    fig.suptitle(r"Comparison of $tSNR$ maps")
    
    for i, img in enumerate(images, start=1):
        img_run = img[run_idx]
        
        tsnr_map = _compute_tsnr_map(img_run)
        vmax = np.percentile(tsnr_map, 99)
        
        
        

        # display = plotting.plot_stat_map(
        #     tsnr_map,
        #     display_mode="z",
        #     cut_coords=[-30, -8, 13, 30],
        #     cmap="magma",
        #     colorbar=True,
        #     threshold=0,
        #     vmin=0,
        #     vmax=vmax,
        #     black_bg=True,
        #     draw_cross=False,
        #     title=f"tSNR - {model_name}, run {i:02d}"
        # )

        # output_path = PLOTS_DIR / f"tsnr_{model_name}_run{i:02d}.png"
        # display.savefig(output_path, dpi=200, bbox_inches="tight")
        # display.close()
