"""
Visualize inference results by plotting slices at the predicted peak and ground truth.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_npz(path: Path):
    return np.load(path)

def plot_slices(vol, prob, z_idx, title_prefix, output_path):
    """Plot axial, coronal, and sagittal slices at a given Z index."""
    # Find max prob location in this Z slice to center X/Y
    # If z_idx is scalar, we take that slice.
    
    # We need the full 3D coordinate to center the crosshairs
    # Let's find the max prob in the entire volume to get X/Y if not provided?
    # Or just find max in this slice.
    
    slice_prob = prob[z_idx, :, :]
    y_idx, x_idx = np.unravel_index(np.argmax(slice_prob), slice_prob.shape)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Axial (XY)
    axes[0, 0].imshow(vol[z_idx, :, :], cmap="gray")
    axes[0, 0].set_title(f"{title_prefix} Axial Vol (Z={z_idx})")
    axes[0, 0].scatter(x_idx, y_idx, c='r', s=10)
    
    axes[1, 0].imshow(prob[z_idx, :, :], cmap="jet", vmin=0, vmax=1)
    axes[1, 0].set_title(f"{title_prefix} Axial Prob (Z={z_idx})")
    
    # Coronal (XZ) - using y_idx
    axes[0, 1].imshow(vol[:, y_idx, :], cmap="gray", aspect='auto') # Z is 0-axis (vertical in imshow usually?)
    # usually imshow shows (row, col). vol is (Z, Y, X). 
    # vol[:, y, :] is (Z, X). 
    axes[0, 1].set_title(f"{title_prefix} Coronal Vol (Y={y_idx})")
    axes[0, 1].axhline(z_idx, c='r')
    axes[0, 1].axvline(x_idx, c='r')

    axes[1, 1].imshow(prob[:, y_idx, :], cmap="jet", vmin=0, vmax=1, aspect='auto')
    axes[1, 1].set_title(f"{title_prefix} Coronal Prob (Y={y_idx})")
    
    # Sagittal (YZ) - using x_idx
    axes[0, 2].imshow(vol[:, :, x_idx], cmap="gray", aspect='auto') # (Z, Y)
    axes[0, 2].set_title(f"{title_prefix} Sagittal Vol (X={x_idx})")
    axes[0, 2].axhline(z_idx, c='r')
    axes[0, 2].axvline(y_idx, c='r')

    axes[1, 2].imshow(prob[:, :, x_idx], cmap="jet", vmin=0, vmax=1, aspect='auto')
    axes[1, 2].set_title(f"{title_prefix} Sagittal Prob (X={x_idx})")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved {output_path}")
    plt.close()

CONFIG = {
    "npz_path": "cloud_models/MihaiB-dev/new-loss-2_curriculum-learning-alpha-5/inference_test.npz",
    "output_dir": "cloud_models/MihaiB-dev/new-loss-2_curriculum-learning-alpha-5/visualizations",
    "gt_z": 191,
    "pred_z": 224
}

def main():
    args = CONFIG
    
    data = np.load(args["npz_path"])
    vol = data["vol"]
    prob = data["prob"]
    
    out_dir = Path(args["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args["pred_z"] is not None:
        plot_slices(vol, prob, args["pred_z"], "Pred", out_dir / "pred_peak.png")
        
    if args["gt_z"] is not None:
        plot_slices(vol, prob, args["gt_z"], "GT", out_dir / "gt_loc.png")

if __name__ == "__main__":
    main()
