# Source - https://stackoverflow.com/a/68907282
# Posted by fepegar
# Retrieved 2026-04-15, License - CC BY-SA 4.0

from pathlib import Path
import numpy as np
import nibabel as nib
from skimage import io
import imageio.v2 as imageio


def to_uint8(data):
    data -= data.min()
    data /= data.max()
    data *= 255
    return data.astype(np.uint8)


def nii_to_jpgs(input_path, rgb=False):
    output_dir = Path(input_path).parent / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = nib.load(input_path).get_fdata()

    if data.ndim == 3:
        data = data[..., np.newaxis]  # add fake channel dim

    num_slices = data.shape[2]
    num_channels = data.shape[3]

    for channel in range(num_channels):
        volume = data[..., channel]
        volume = to_uint8(volume)

        channel_dir = output_dir / f'channel_{channel}'
        channel_dir.mkdir(exist_ok=True, parents=True)

        for slice_idx in range(num_slices):
            slice_data = volume[:, :, slice_idx]

            if rgb:
                slice_data = np.stack([slice_data]*3, axis=-1)

            output_path = channel_dir / f'channel_{channel}_slice_{slice_idx}.jpg'
            io.imsave(output_path, slice_data)



def gif_from_same_slice(root_dir, slice_idx, output_path, duration=0.2):
    """
    Create GIF across channels for a fixed slice index.
    """
    root = Path(root_dir)
    frames = []

    channel_dirs = sorted(root.glob("channel_*"))

    for ch_dir in channel_dirs:
        # extract channel number
        ch = ch_dir.name.split("_")[-1]
        img_path = ch_dir / f"channel_{ch}_slice_{slice_idx}.jpg"

        if not img_path.exists():
            print(f"Missing: {img_path}")
            continue

        frames.append(imageio.imread(img_path))

    if frames:
        imageio.mimsave(output_path, frames, duration=duration)
        print(f"Saved: {output_path}")
    else:
        print("No frames found.")


def gif_from_channel_all_slices(root_dir, channel_idx, output_path, duration=0.2):
    """
    Create GIF from all slices of a specific channel.
    """
    root = Path(root_dir)
    ch_dir = root / f"channel_{channel_idx}"

    if not ch_dir.exists():
        print(f"Channel dir not found: {ch_dir}")
        return

    # sort by slice index
    images = sorted(
        ch_dir.glob(f"channel_{channel_idx}_slice_*.jpg"),
        key=lambda p: int(p.stem.split("_")[-1])
    )

    frames = [imageio.imread(img) for img in images]

    if frames:
        imageio.mimsave(output_path, frames, duration=duration)
        print(f"Saved: {output_path}")
    else:
        print("No frames found.")


# ======================
# 🔧 USAGE EXAMPLES
# ======================



if __name__ == "__main__":
    input_path = "ds000003_R2.0.2/sub-01/func/sub-01_task-rhymejudgment_bold.nii.gz"
    # nii_to_jpgs(input_path)
    
    root_dir = Path(input_path).parent / "images"
    print(root_dir)

    # 1. GIF across channels for slice 50
    gif_from_same_slice(
        root_dir,
        slice_idx=0,
        output_path="slice_50_across_channels.gif"
    )

    # 2. GIF for channel 0 across all slices
    gif_from_channel_all_slices(
        root_dir,
        channel_idx=0,
        output_path="channel_0_all_slices.gif"
    )