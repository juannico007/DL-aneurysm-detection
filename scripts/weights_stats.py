"""
Activation/gradient stats for a trained UNet, similar to per_layer_stats_visualization.

It loads one patch from patches.csv (first positive if available), runs a forward/backward
with a simple Dice + BCE loss, and plots per-layer histograms of activations and gradients.

Configure HYPERPARAMS below, then run:
    python3 scripts/weights_stats.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from model.unet import UNet  # noqa: E402


HYPERPARAMS: Dict[str, object] = {
    "h5_path": Path("h5-aneurysm.h5"),
    "patch_csv": Path("patches/patches.csv"),
    "checkpoint": Path("cloud_models/MihaiB-dev/patches-3d-unet_002/MihaiB-dev_patches-3d-unet_002.pt"),
    "hyperparams_json": Path("cloud_models/MihaiB-dev/patches-3d-unet_002/hyperparameters.json"),
    "unet_kwargs": {},
    "radius": 5.0,
    "patch_size": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": Path("runs/weight_stats"),
}


@dataclass
class StatSummary:
    name: str
    mean: float
    std: float
    dtype: str


def make_sphere_mask(shape: Sequence[int], center: Sequence[float], radius: float) -> np.ndarray:
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz, cy, cx = center
    dist2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return (dist2 <= radius**2).astype(np.uint8)


def load_first_patch(
    h5_path: Path, patch_csv: Path, patch_size: int, radius: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load the first positive patch (else first row) and its GT mask."""
    with patch_csv.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No rows in patches CSV.")
    try:
        row = next(r for r in rows if int(r["label"]) == 1)
    except StopIteration:
        row = rows[0]

    sid = row["series_id"]
    cx = float(row["center_x"])
    cy = float(row["center_y"])
    cz = float(row["center_z"])
    ax = row.get("aneurysm_x")
    ay = row.get("aneurysm_y")
    az = row.get("aneurysm_z")
    ax = float(ax) if ax else cx
    ay = float(ay) if ay else cy
    az = float(az) if az else cz

    with h5py.File(h5_path, "r") as handle:
        vol = handle["series"][sid]["vol"]
        half = patch_size / 2.0
        z0 = int(max(0, min(vol.shape[0] - patch_size, round(cz - half))))
        y0 = int(max(0, min(vol.shape[1] - patch_size, round(cy - half))))
        x0 = int(max(0, min(vol.shape[2] - patch_size, round(cx - half))))
        z1, y1, x1 = z0 + patch_size, y0 + patch_size, x0 + patch_size
        patch_np = vol[z0:z1, y0:y1, x0:x1]
    if patch_np.shape != (patch_size, patch_size, patch_size):
        pad_z = patch_size - patch_np.shape[0]
        pad_y = patch_size - patch_np.shape[1]
        pad_x = patch_size - patch_np.shape[2]
        patch_np = np.pad(patch_np, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")

    mask_np = make_sphere_mask(
        patch_np.shape,
        (az - z0, ay - y0, ax - x0),
        radius=radius,
    )
    patch_t = torch.from_numpy(patch_np).unsqueeze(0).unsqueeze(0).float()
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()
    return patch_t, mask_t


def load_unet_from_config() -> UNet:
    p = HYPERPARAMS
    unet_kwargs = dict(p.get("unet_kwargs") or {})
    hp_json = p.get("hyperparams_json")
    if hp_json:
        hp_path = Path(hp_json)
        if hp_path.exists():
            with hp_path.open() as f:
                hp_data = json.load(f)
            if isinstance(hp_data, dict) and "unet" in hp_data:
                unet_kwargs.update(hp_data["unet"])
                print(f"[Info] Loaded UNet args from {hp_path}")
    return UNet(**unet_kwargs)


class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1.0):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


def get_layer_stats(tensors: List[torch.Tensor], absolute: bool = False):
    means, stds = [], []
    for t in tensors:
        t = t.detach().float()
        if absolute:
            t = t.abs()
        means.append(float(t.mean().item()))
        stds.append(float(t.std().item()))
    return means, stds


def plot_hist(tensors: List[torch.Tensor], names: List[str], outfile: Path, means=None, stds=None):
    n = len(tensors)
    ncols = min(6, max(1, int(np.ceil(np.sqrt(n)))))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for idx, (t, ax) in enumerate(zip(tensors, axes)):
        arr = t.detach().cpu().numpy().flatten()
        ax.hist(arr, bins=20)
        title = names[idx]
        if means is not None and stds is not None:
            title += f"\nmean {means[idx]:.4f}\nstd {stds[idx]:.4f}"
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
    for ax in axes[len(tensors):]:
        ax.axis("off")
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def main():
    p = HYPERPARAMS
    device = torch.device(p["device"])
    model = load_unet_from_config().to(device)
    state = torch.load(Path(p["checkpoint"]), map_location="cpu")
    model.load_state_dict(state, strict=False)

    # Collect activations and gradients via hooks
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []
    names: List[str] = []

    def save_activation(name):
        def hook(module, inp, out):
            activations.append(out.detach().cpu())
            names.append(name)
            if out.requires_grad:
                out.register_hook(lambda grad: gradients.append(grad.detach().cpu()))
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d, nn.Linear)):
            module.register_forward_hook(save_activation(name))

    patch_t, mask_t = load_first_patch(
        Path(p["h5_path"]),
        Path(p["patch_csv"]),
        patch_size=int(p["patch_size"]),
        radius=float(p["radius"]),
    )
    patch_t = patch_t.to(device, dtype=torch.float32)
    mask_t = mask_t.to(device, dtype=torch.float32)

    model.train()  # ensure we can backprop; BN will use batch stats
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    out_seg, out_cls = model(patch_t)
    cls_target = torch.tensor([[1.0]], device=device, dtype=torch.float32)  # positive label
    loss = dice(out_seg, mask_t) + bce(out_cls, cls_target)
    model.zero_grad()
    loss.backward()

    # Compute stats and plot
    out_dir = Path(p["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    act_means, act_stds = get_layer_stats(activations, absolute=False)
    grad_means, grad_stds = get_layer_stats(gradients, absolute=True)

    plot_hist(activations, names, out_dir / "activations_hist.png", act_means, act_stds)
    plot_hist(gradients, names[: len(gradients)], out_dir / "gradients_hist.png", grad_means, grad_stds)

    summary = {
        "activation_mean": act_means,
        "activation_std": act_stds,
        "gradient_mean_abs": grad_means,
        "gradient_std_abs": grad_stds,
        "layers": names,
        "loss": float(loss.item()),
    }
    with (out_dir / "stats.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Done] Saved activation/gradient hists to {out_dir}")


if __name__ == "__main__":
    main()
