"""
Per-layer activation and gradient histograms for the patch-based UNet checkpoint.

This mirrors the style of per_layer_stats_visualization: run one batch forward/backward,
gather activations and gradients, compute mean/std, and plot histograms.

Configure HYPERPARAMS below, then run:
    python3 scripts/per_layer_stats_patches.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from model.unet import UNet  # noqa: E402
from model.training_pipeline import (
    load_patch_records,
    PatchAneurysmDataset,
    DiceLoss,
)


HYPERPARAMS: Dict[str, object] = {
    "h5_path": Path("h5-aneurysm.h5"),
    "patch_csv": Path("patches/patches.csv"),
    "checkpoint": Path("cloud_models/MihaiB-dev/patches-3d-unet_002/MihaiB-dev_patches-3d-unet_002.pt"),
    "hyperparams_json": Path("cloud_models/MihaiB-dev/patches-3d-unet_002/hyperparameters.json"),
    "unet_kwargs": {},
    "radius": 5.0,
    "patch_size": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": Path("runs/per_layer_stats"),
}


def get_layer_stats(tensors: List[torch.Tensor], absolute: bool = False) -> Tuple[List[float], List[float]]:
    avg, std = [], []
    for t in tensors:
        t_det = t.detach().float()
        if absolute:
            t_det = t_det.abs()
        avg.append(float(t_det.mean().item()))
        std.append(float(t_det.std().item()))
    return avg, std


def plot_hist(hs: List[torch.Tensor], names: List[str], outfile: Path, xrange=None, avg=None, sd=None):
    n = len(hs)
    ncols = min(6, max(1, int(np.ceil(np.sqrt(n)))))
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(3 * ncols, 3 * nrows))
    for layer in range(n):
        plt.subplot(nrows, ncols, layer + 1)
        activations = hs[layer].detach().cpu().numpy().flatten()
        plt.hist(activations, bins=20, range=xrange)
        title = f"{names[layer]}"
        if avg:
            title += f"\nmean {avg[layer]:.4f}"
        if sd:
            title += f"\nstd {sd[layer]:.4f}"
        plt.title(title, fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close()


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


def main():
    p = HYPERPARAMS
    device = torch.device(p["device"])
    model = load_unet_from_config().to(device)
    state = torch.load(Path(p["checkpoint"]), map_location="cpu")
    model.load_state_dict(state, strict=False)

    # Load one batch (prefer positive) from patches
    records = load_patch_records(Path(p["patch_csv"]))
    if not records:
        raise ValueError("No patch records found.")
    try:
        first_pos = next(i for i, r in enumerate(records) if r.label == 1)
    except StopIteration:
        first_pos = 0
    records = [records[first_pos]]

    dataset = PatchAneurysmDataset(
        h5_path=Path(p["h5_path"]),
        records=records,
        patch_size=int(p["patch_size"]),
        transform=None,
        radius=float(p["radius"]),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []
    layer_names: List[str] = []

    def get_layer_data(model):
        grads = []
        names = []
        acts = model.activations
        for name, param in model.named_parameters():
            if param.requires_grad and name.endswith('.weight') and ("conv" in name or "final" in name or "class" in name):
                names.append(name)
                if param.grad is None:
                    grads.append(torch.zeros(1, 1, 1, 1, 1))
                else:
                    grads.append(param.grad)
        return names, acts, grads

    model.train()
    for x_batch, mask_batch, y_batch in loader:
        x_batch = x_batch.to(device, dtype=torch.float32)
        mask_batch = mask_batch.to(device, dtype=torch.float32)
        y_batch = y_batch.float().unsqueeze(1).to(device, dtype=torch.float32)
        out_seg, out_cls = model(x_batch)
        loss = dice(out_seg, mask_batch) + bce(out_cls, y_batch)
        model.zero_grad()
        loss.backward()
        layer_names, activations, gradients = get_layer_data(model)
        break

    activation_mean, activation_std = get_layer_stats(activations)
    gradient_mean, gradient_std = get_layer_stats(gradients, absolute=True)

    out_dir = Path(p["output_dir"])
    plot_hist(gradients, layer_names, out_dir / "gradients_hist.png", xrange=None, avg=gradient_mean, sd=gradient_std)
    plot_hist(activations, [f"act_{i}" for i in range(len(activations))], out_dir / "activations_hist.png", xrange=None, avg=activation_mean, sd=activation_std)

    summary = {
        "activation_mean": activation_mean,
        "activation_std": activation_std,
        "gradient_mean_abs": gradient_mean,
        "gradient_std_abs": gradient_std,
        "layers": layer_names,
        "loss": float(loss.item()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "stats.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Done] Saved activation/gradient histograms to {out_dir}")


if __name__ == "__main__":
    main()
