"""
Sliding-window inference for a trained 3D U-Net on a single volume.

Configuration (edit HYPERPARAMS below):
  - h5_path: path to HDF5 with series/vol
  - series_id: which volume to run on
  - checkpoint: path to trained .pt
  - stride: sliding window stride (voxels)
  - patch_size: size of cubic patch (voxels)
  - radius: sphere radius for optional sphere mask (for debugging)
  - threshold: probability threshold for mask binarization
  - output_prefix: base path for outputs (npz, nifti optional)

Outputs:
  - <output_prefix>.npz containing:
      prob: (D,H,W) float16 probability map
      mask: (D,H,W) uint8 binary mask (thresholded)
      volume: (D,H,W) float16 input volume
      meta: JSON string with settings

Run:
  python3 scripts/sliding_inference.py
"""

from __future__ import annotations
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Sequence

import csv
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from model.unet import UNet  # noqa: E402


HYPERPARAMS: Dict[str, object] = {
    "h5_path": Path("src/train_dataset.h5"),
    "checkpoint": Path("cloud_models/MihaiB-dev/suppression-loss_002/MihaiB-dev_suppression-loss_002.pt"),
    "localizers": None,  # optional; builds GT sphere mask
    "radius": 5.0,  # sphere radius for GT mask
    # Optional: path to hyperparameters.json (will use its "unet" block if present)
    "hyperparams_json":  Path("cloud_models/MihaiB-dev/suppression-loss_002/hyperparameters.json"),
    # Optional: override UNet args directly to match training checkpoint
    "unet_kwargs": {
        # e.g., "start_filters": 16, "out_channels": 1, "normalization": "group8"
    },
    "patch_size": 64,
    "stride": 24,
    "threshold": 0.5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_prefix": Path("runs/random_shit"),
    "use_mixed_precision": False,
    "save_nifti": True,
}


@dataclass
class Meta:
    series_id: str
    h5_path: str
    checkpoint: str
    patch_size: int
    stride: int
    threshold: float
    device: str
    radius: float
    localizers: str


def sliding_window(volume: np.ndarray, patch_size: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """Yield patches with their start indices."""
    D, H, W = volume.shape
    z_starts = list(range(0, max(D - patch_size, 0) + 1, stride))
    y_starts = list(range(0, max(H - patch_size, 0) + 1, stride))
    x_starts = list(range(0, max(W - patch_size, 0) + 1, stride))
    if z_starts[-1] != D - patch_size:
        z_starts.append(D - patch_size)
    if y_starts[-1] != H - patch_size:
        y_starts.append(H - patch_size)
    if x_starts[-1] != W - patch_size:
        x_starts.append(W - patch_size)
    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                patch = volume[z : z + patch_size, y : y + patch_size, x : x + patch_size]
                yield patch, (z, y, x)


def load_localizer_points(csv_path: Path) -> Dict[str, List[Tuple[float, float, float]]]:
    """Return mapping series_id -> list of (z, y, x) points."""
    out: Dict[str, List[Tuple[float, float, float]]] = {}
    if csv_path is None or not csv_path.exists():
        return out
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        required = {"SeriesInstanceUID", "x_new", "y_new", "z_new"}
        if not required.issubset(reader.fieldnames or []):
            return out
        for row in reader:
            try:
                z = float(row["z_new"])
                y = float(row["y_new"])
                x = float(row["x_new"])
            except (TypeError, ValueError):
                continue
            out.setdefault(row["SeriesInstanceUID"], []).append((z, y, x))
    return out


def make_sphere_mask(shape: Sequence[int], center: Sequence[float], radius: float) -> np.ndarray:
    """Binary sphere mask in a given volume shape."""
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz, cy, cx = center
    dist2 = np.square(x - cx)+ np.square(y - cy) + np.square(z - cz)
    return (dist2 <= radius**2).astype(np.uint8)


def run_inference(series_id, threshold, device, predicting_with_mask = False):
    p = HYPERPARAMS
    patch_size = int(p["patch_size"])
    stride = int(p["stride"])
    out_prefix = Path(p["output_prefix"])
    #Load data
    with h5py.File(h5_path, "r") as f:
        vol = f["series"][series_id]["vol"][:]
    vol = np.asarray(vol, dtype=np.float32)
    print(vol.shape, vol.mean())
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

    # Build GT mask from localizers if provided
    gt_mask = None
    loc_path = p.get("localizers")
    if loc_path:
        loc_points = load_localizer_points(Path(loc_path))
        pts = loc_points.get(series_id, [])
        if pts:
            rad = float(p.get("radius", 5.0))
            gt_mask = np.zeros_like(vol, dtype=np.uint8)
            for c in pts:
                gt_mask |= make_sphere_mask(vol.shape, c, radius=rad)

    prob = np.zeros_like(vol, dtype=np.float32)
    counts = np.zeros_like(vol, dtype=np.float32)

    use_amp = bool(p.get("use_mixed_precision", False) and device.type == "cuda")
    final_class_output = False
    with torch.no_grad():
        for patch_np, (z, y, x) in sliding_window(vol, patch_size=patch_size, stride=stride):
            patch_t = torch.from_numpy(patch_np).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    seg_logits, class_output = model(patch_t)
            else:
                seg_logits, class_output = model(patch_t)
            class_output = torch.sigmoid(class_output)
            sig = class_output
            class_output = (class_output >= threshold)
            #print(class_output, sig)
            final_class_output = final_class_output or class_output
            seg_probs = torch.sigmoid(seg_logits.float()).cpu().numpy()[0, 0]
            prob[z : z + patch_size, y : y + patch_size, x : x + patch_size] += seg_probs
            counts[z : z + patch_size, y : y + patch_size, x : x + patch_size] += 1.0

    counts[counts == 0] = 1.0   
    prob /= counts
    mask = (prob >= threshold).astype(np.uint8)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    meta = Meta(
        series_id=series_id,
        h5_path=str(h5_path),
        checkpoint=str(ckpt),
        patch_size=patch_size,
        stride=stride,
        threshold=threshold,
        device=str(device),
        radius=float(p.get("radius", 5.0)),
        localizers=str(loc_path) if loc_path else "",
    )
    np.savez_compressed(
        out_prefix.with_suffix(".npz"),
        prob=prob,
        mask=mask,
        volume=vol,
        gt_mask=gt_mask if gt_mask is not None else np.zeros((1, 1, 1), dtype=np.uint8),
        metadata=json.dumps(asdict(meta)),
    )
    if p.get("save_nifti", False):
        try:
            import nibabel as nib
            affine = np.eye(4)
            nib.save(nib.Nifti1Image(prob.astype(np.float32), affine), out_prefix.with_suffix(".prob.nii.gz"))
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), out_prefix.with_suffix(".mask.nii.gz"))
            nib.save(nib.Nifti1Image(vol.astype(np.float32), affine), out_prefix.with_suffix(".vol.nii.gz"))
            if gt_mask is not None:
                nib.save(nib.Nifti1Image(gt_mask.astype(np.uint8), affine), out_prefix.with_suffix(".gtmask.nii.gz"))
            #print(f"[Done] Saved NIfTI volumes to {out_prefix}.prob.nii.gz / .mask.nii.gz / .vol.nii.gz")
        except ImportError:
            pass
            #print("[Warning] nibabel not installed; skipping NIfTI export.")
    #print(f"[Done] Saved {out_prefix.with_suffix('.npz')}, prob mean {prob.mean():.4f}, mask sum {mask.sum()}")
    if predicting_with_mask:
        return mask.max()
    else:
        print(final_class_output)
        exit()
        return int(final_class_output)


if __name__ == "__main__":
    threshold = float(HYPERPARAMS["threshold"])
    h5_path = Path(HYPERPARAMS["h5_path"])
    ckpt = Path(HYPERPARAMS["checkpoint"])
    unet_kwargs = dict(HYPERPARAMS.get("unet_kwargs") or {})
    hp_json = HYPERPARAMS.get("hyperparams_json")
    device = torch.device(HYPERPARAMS["device"])

    
    train_df = pd.read_csv("ct_preprocessed/train.csv")
    series_id = train_df["SeriesInstanceUID"].iloc[0]
    
    #Load model
    # Build UNet with provided kwargs or from hyperparams_json if available
    if hp_json:
        hp_path = Path(hp_json)
        if hp_path.exists():
            with hp_path.open() as f:
                hp_data = json.load(f)
            if isinstance(hp_data, dict) and "unet" in hp_data:
                unet_kwargs.update(hp_data["unet"])
                print(f"[Info] Loaded UNet args from {hp_path}")
    model = UNet(**unet_kwargs)
    state = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[Warning] load_state_dict(strict=False). Missing: {missing}, Unexpected: {unexpected}")
    model.to(device)
    model.eval()

    predictions = []
    labels = []
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    for index, row in tqdm(train_df[["SeriesInstanceUID", "Aneurysm Present"]].iloc[:10].iterrows()):
        x = row[0]
        print(x)
        y = row[1]
        print("Ground truth:", y)
        prediction = run_inference(series_id, threshold, device)
        
        predictions.append(prediction)
        labels.append(y)

    cm = confusion_matrix(labels, predictions)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig("confusion_matrix_higher_threshold.png")
#1.2.826.0.1.3680043.8.498.82300188681258136694033841609727559375
#1.2.826.0.1.3680043.8.498.98763049464685085039537379330934967318