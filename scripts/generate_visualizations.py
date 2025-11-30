"""
Generate training metric plots and example inference outputs.

This script is intentionally decomposed into reusable functions:
  - load history and plot metrics
  - pick positive/negative series ids
  - run sliding-window inference with optional blending
  - plot layer activation/gradient histograms (if provided)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError as e:
    print(f"[Warn] matplotlib unavailable: {e}. Plots will be skipped.")
    _HAS_MPL = False
import numpy as np
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from model.unet import UNet
from model.training_pipeline import make_heatmaps  # reuse heatmap generator


# -----------------------------
# Data classes and configuration
# -----------------------------

@dataclass
class HistoryPaths:
    history: Path
    output_folder: Path


@dataclass
class InferenceConfig:
    h5_path: Path
    checkpoint: Path
    patch_size: int = 64
    stride: int = 24
    threshold: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    blend_mode: str = "hann"  # {"hann", "gaussian", "naive"}
    gaussian_sigma: Optional[float] = None
    use_mixed_precision: bool = False
    localizers_path: Optional[Path] = None
    gauss_sigma: float = 15.0
    gauss_tau: float = 0.0


# -----------------------------
# General utilities
# -----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_history(history_path: Path) -> Dict[str, Iterable[float]]:
    print(f"[Info] Loading history from {history_path}")
    with history_path.open("rb") as handle:
        return pickle.load(handle)


def plot_metric(history: Dict[str, Iterable[float]], metric: str, output_folder: Path) -> None:
    if not _HAS_MPL:
        return
    fig = plt.figure()
    plt.plot(history.get(f"train_{metric}", []), label=f"train {metric}")
    plt.plot(history.get(f"val_{metric}", []), label=f"val {metric}")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    output_path = output_folder / f"visualizations/{metric}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_all_metrics(history: Dict[str, Iterable[float]], output_folder: Path) -> None:
    if not _HAS_MPL:
        print("[Warn] Skipping metric plots (matplotlib missing).")
        return
    ensure_dir(output_folder / "visualizations")
    for key in history.keys():
        if key.startswith("train"):
            metric = key[len("train_") :]
            plot_metric(history, metric, output_folder)


# -----------------------------
# Sliding inference helpers
# -----------------------------

def sliding_window(volume: np.ndarray, patch_size: int, stride: int) -> Iterable[Tuple[np.ndarray, Tuple[int, int, int]]]:
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


def build_weight_mask(patch_size: int, mode: str = "hann", sigma: Optional[float] = None) -> np.ndarray:
    mode = (mode or "hann").lower()
    if mode == "naive":
        return np.ones((patch_size, patch_size, patch_size), dtype=np.float32)
    if mode == "hann":
        h = np.hanning(patch_size)
        return (h[:, None, None] * h[None, :, None] * h[None, None, :]).astype(np.float32)
    if mode == "gaussian":
        if sigma is None:
            sigma = patch_size / 6.0
        coords = np.linspace(-(patch_size - 1) / 2.0, (patch_size - 1) / 2.0, patch_size)
        zz, yy, xx = np.meshgrid(coords, coords, coords, indexing="ij")
        mask = np.exp(-(xx**2 + yy**2 + zz**2) / (2.0 * sigma * sigma))
        mask -= mask.min()
        vmax = mask.max()
        if vmax > 0:
            mask /= vmax
        return mask.astype(np.float32)
    raise ValueError(f"Unsupported blend_mode '{mode}'. Use 'naive', 'hann', or 'gaussian'.")


def load_unet_from_config(hp_json: Path, override_kwargs: Optional[Dict[str, object]] = None) -> UNet:
    unet_kwargs: Dict[str, object] = {}
    if hp_json.exists():
        with hp_json.open() as f:
            hp_data = json.load(f)
        if isinstance(hp_data, dict) and "unet" in hp_data:
            unet_kwargs.update(hp_data["unet"])
    if override_kwargs:
        unet_kwargs.update(override_kwargs)
    return UNet(**unet_kwargs)


def build_gaussian_gt(shape: Optional[Tuple[int, int, int]], series_id: str, loc_path: Optional[Path], sigma: float, tau: float) -> Optional[np.ndarray]:
    """Construct a Gaussian GT mask from localizer CSV for a given series_id."""
    if not loc_path or not Path(loc_path).exists():
        return None
    if shape is None:
        try:
            with h5py.File(Path("train_dataset.h5"), "r") as f:
                shape = f["series"][series_id]["vol"].shape
        except Exception:
            return None
    try:
        import csv
        pts = []
        with Path(loc_path).open() as f:
            reader = csv.DictReader(f)
            required = {"SeriesInstanceUID", "x_new", "y_new", "z_new"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("localizers CSV missing required columns")
            for row in reader:
                if row.get("SeriesInstanceUID") != series_id:
                    continue
                try:
                    z = float(row["z_new"])
                    y = float(row["y_new"])
                    x = float(row["x_new"])
                    pts.append((z, y, x))
                except (TypeError, ValueError):
                    continue
        if not pts:
            return None
        heatmap = np.zeros(shape, dtype=np.float32)
        for cz, cy, cx in pts:
            heatmap += make_heatmaps(shape, (cz, cy, cx), [sigma], tau_gauss=tau)[0]
        return np.clip(heatmap, 0.0, 1.0)
    except Exception as e:
        print(f"[Warn] Failed to build Gaussian GT mask: {e}")
        return None


def run_sliding_inference(cfg: InferenceConfig, series_id: str, out_prefix: Path, gt_gauss: Optional[np.ndarray] = None) -> None:
    print(f"[Info] Running inference for {series_id} -> {out_prefix}")
    with h5py.File(cfg.h5_path, "r") as f:
        vol = f["series"][series_id]["vol"][:]
    vol = np.nan_to_num(np.asarray(vol, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    model = load_unet_from_config(cfg.checkpoint.parent / "hyperparameters.json")
    print(f"[Info] Loading checkpoint {cfg.checkpoint}")
    state = torch.load(cfg.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    device = torch.device(cfg.device)
    model.to(device)
    model.eval()

    prob = np.zeros_like(vol, dtype=np.float32)
    counts = np.zeros_like(vol, dtype=np.float32)
    weight_mask = build_weight_mask(cfg.patch_size, mode=cfg.blend_mode, sigma=cfg.gaussian_sigma)
    
    attention_maps = []
    for _ in range(len(model.up_blocks)):
        attention_maps.append(np.zeros_like(vol, dtype=np.float32))
        
    attention_counts = [np.zeros_like(vol, dtype=np.float32) for _ in range(len(model.up_blocks))]

    use_amp = bool(cfg.use_mixed_precision and device.type == "cuda")

    with torch.no_grad():
        for patch_np, (z, y, x) in sliding_window(vol, patch_size=cfg.patch_size, stride=cfg.stride):
            patch_t = torch.from_numpy(patch_np).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    seg_logits, _ = model(patch_t)
            else:
                seg_logits, _ = model(patch_t)
            seg_probs = torch.sigmoid(seg_logits.float()).cpu().numpy()[0, 0]
            prob[z : z + cfg.patch_size, y : y + cfg.patch_size, x : x + cfg.patch_size] += seg_probs * weight_mask
            counts[z : z + cfg.patch_size, y : y + cfg.patch_size, x : x + cfg.patch_size] += weight_mask
            for level_idx, up_block in enumerate(model.up_blocks):
                if hasattr(up_block.attention, 'last_attention'):
                    att = up_block.attention.last_attention
                    if att is not None:
                        att_resized = torch.nn.functional.interpolate(
                            att, 
                            size=(cfg.patch_size, cfg.patch_size, cfg.patch_size),
                            mode='trilinear',
                            align_corners=False
                        ).cpu().numpy()[0, 0]
                        
                        attention_maps[level_idx][
                            z : z + cfg.patch_size, 
                            y : y + cfg.patch_size, 
                            x : x + cfg.patch_size
                        ] += att_resized * weight_mask
                        
                        attention_counts[level_idx][
                            z : z + cfg.patch_size, 
                            y : y + cfg.patch_size, 
                            x : x + cfg.patch_size
                        ] += weight_mask

    counts[counts == 0] = 1.0
    prob /= counts
    mask = (prob >= cfg.threshold).astype(np.uint8)
    for level_idx in range(len(attention_maps)):
        attention_counts[level_idx][attention_counts[level_idx] == 0] = 1.0
        attention_maps[level_idx] /= attention_counts[level_idx]

    ensure_dir(out_prefix.parent)
    cfg_serializable = json.loads(json.dumps(asdict(cfg), default=str))
    save_dict = {
        'prob': prob.astype(np.float16),
        'mask': mask.astype(np.uint8),
        'gt_gauss': gt_gauss.astype(np.float32) if gt_gauss is not None else np.zeros((1, 1, 1), dtype=np.float32),
        'metadata': json.dumps(cfg_serializable),
        'series_id': series_id,
    }
    
    np.savez_compressed(out_prefix.with_suffix(".npz"), **save_dict)
    print(f"[Info] Saved inference outputs to {out_prefix.with_suffix('.npz')}")
    
    # Save NIfTI volumes for easier inspection
    try:
        import nibabel as nib
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(prob.astype(np.float32), affine), out_prefix.with_suffix(".prob.nii.gz"))
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), out_prefix.with_suffix(".mask.nii.gz"))
        nib.save(nib.Nifti1Image(vol.astype(np.float32), affine), out_prefix.with_suffix(".vol.nii.gz"))
        if gt_gauss is not None:
            nib.save(nib.Nifti1Image(gt_gauss.astype(np.float32), affine), out_prefix.with_suffix(".gt_gauss.nii.gz"))
        for level_idx, att_map in enumerate(attention_maps):
            att_path = out_prefix.with_name(f"{out_prefix.stem}.attention_L{level_idx}.nii.gz")
            nib.save(nib.Nifti1Image(att_map.astype(np.float32), affine), att_path)
            print(f"[Info] Saved attention map level {level_idx}")
        print(f"[Info] Saved NIfTI outputs (.prob/.mask/.vol) for {series_id}")
    except ImportError:
        print("[Warn] nibabel not available; skipping NIfTI export.")


def pick_series_ids(patch_csv: Path, pos_label: int = 1, neg_label: int = 0) -> Tuple[str, str]:
    import csv
    pos_ids, neg_ids = [], []
    with patch_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lbl = int(row.get("label", ""))
            except (TypeError, ValueError):
                continue
            sid = row.get("series_id")
            if not sid:
                continue
            if lbl == pos_label:
                pos_ids.append(sid)
            elif lbl == neg_label:
                neg_ids.append(sid)
    print(f"[Info] Found {len(pos_ids)} positive and {len(neg_ids)} negative series IDs in {patch_csv}")
    if not pos_ids or not neg_ids:
        raise ValueError("Could not find both a positive and negative series_id in patch CSV.")
    return pos_ids[0], neg_ids[0]


def pick_series_from_train(train_csv: Path, pos_col: str = "Aneurysm Present") -> Tuple[str, str]:
    import csv
    pos_ids, neg_ids = [], []
    with train_csv.open() as f:
        reader = csv.DictReader(f)
        if "SeriesInstanceUID" not in reader.fieldnames or pos_col not in reader.fieldnames:
            raise ValueError("train.csv missing required columns (SeriesInstanceUID, Aneurysm Present).")
        for row in reader:
            try:
                val = float(row.get(pos_col, 0))
            except (TypeError, ValueError):
                val = 0.0
            sid = row.get("SeriesInstanceUID")
            if not sid:
                continue
            if val > 0:
                pos_ids.append(sid)
            else:
                neg_ids.append(sid)
    print(f"[Info] Found {len(pos_ids)} positive and {len(neg_ids)} negative series IDs in {train_csv}")
    if not pos_ids or not neg_ids:
        raise ValueError("Could not find both positive and negative series IDs in train.csv.")
    return pos_ids[0], neg_ids[0]


# -----------------------------
# Layer statistics helpers
# -----------------------------

def get_layer_stats(x: Sequence[torch.Tensor], absolute: bool = False) -> Tuple[List[float], List[float]]:
    avg, std = [], []
    for layer in x:
        vals = layer.abs() if absolute else layer
        avg.append(vals.mean().detach().cpu().item())
        std.append(vals.std().detach().cpu().item())
    return avg, std


def plot_hist(tensors: Sequence[torch.Tensor], xrange=None, avg: Optional[List[float]] = None, sd: Optional[List[float]] = None, title_prefix: str = "Layer", cols: int = 4, outfile: Optional[Path] = None):
    if not _HAS_MPL:
        return
    rows = int(np.ceil(len(tensors) / cols))
    plt.figure(figsize=(4 * cols, 3 * rows))
    for idx, t in enumerate(tensors):
        plt.subplot(rows, cols, idx + 1)
        vals = t.detach().cpu().numpy().flatten()
        plt.hist(vals, bins=20, range=xrange)
        title = f"{title_prefix} {idx + 1}"
        if avg:
            title += f"\nmean {avg[idx]:.2f}"
        if sd:
            title += f"\nstd {sd[idx]:.4f}"
        plt.title(title)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=220, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_layer_stats(activations: Sequence[torch.Tensor], gradients: Sequence[torch.Tensor], out_dir: Path) -> None:
    if not _HAS_MPL:
        print("[Warn] Skipping layer stat plots (matplotlib missing).")
        return
    ensure_dir(out_dir)
    activation_mean, activation_std = get_layer_stats(activations)
    gradient_mean, gradient_std = get_layer_stats(gradients, absolute=True)
    plot_hist(activations, xrange=None, avg=activation_mean, sd=activation_std, title_prefix="Activation", outfile=out_dir / "activations_hist.png")
    plot_hist(gradients, xrange=None, avg=gradient_mean, sd=gradient_std, title_prefix="Gradient", outfile=out_dir / "gradients_hist.png")


def get_layer_data(model: torch.nn.Module, volume: np.ndarray, patch_size: int = 64) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
    """Run a forward/backward on a random patch to populate activations and gradients."""
    D, H, W = volume.shape
    import random
    z0 = random.randrange(0, max(D - patch_size, 1))
    y0 = random.randrange(0, max(H - patch_size, 1))
    x0 = random.randrange(0, max(W - patch_size, 1))
    patch = volume[z0:z0 + patch_size, y0:y0 + patch_size, x0:x0 + patch_size]
    device = next(model.parameters()).device
    patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    model.train()
    model.zero_grad(set_to_none=True)
    seg_logits, _ = model(patch_t)
    # Use a simple scalar loss to trigger gradients
    loss = seg_logits.mean()
    loss.backward()

    layer_names, grads = [], []
    for name, param in model.named_parameters():
        if param.requires_grad and name.endswith(".weight") and ("conv" in name or "final" in name or "class" in name):
            layer_names.append(name)
            grads.append(param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param))
    activations = getattr(model, "activations", [])
    return layer_names, activations, grads


# -----------------------------
# Main entry
# -----------------------------

def main():
    base_output = Path("cloud_models/RusnacAM/AMR_003")
    history_path = base_output / "history.pickle"
    output_folder = base_output

    # Plot metrics
    try:
        history = load_history(history_path)
        plot_all_metrics(history, output_folder)
    except Exception as e:
        print(f"[Warn] Could not load/plot history: {e}")

    # Sliding inference examples
    try:
        pos_id, neg_id = pick_series_from_train(Path("train.csv"))
    except Exception as e:
        print(f"[Warn] Could not pick positive/negative series IDs for inference examples: {e}")
        pos_id, neg_id = None, None

    ckpt_candidates = list(base_output.glob("40_RusnacAM_AMR_003.pt"))
    if ckpt_candidates and pos_id and neg_id:
        print(f"[Info] Using checkpoint {ckpt_candidates[0]} for inference examples.")
        cfg = InferenceConfig(
            h5_path=Path("train_dataset.h5"),
            checkpoint=ckpt_candidates[0],
            patch_size=64,
            stride=24,
            threshold=0.8,
            device="cuda" if torch.cuda.is_available() else "cpu",
            blend_mode="hann",
            gaussian_sigma=None,
            localizers_path=Path("train_localizers.csv"),
            gauss_sigma=15.0,
            gauss_tau=0.1,
        )
        image_out = base_output / "image_results"
        ensure_dir(image_out)
        gt_pos = build_gaussian_gt(None, pos_id, cfg.localizers_path, cfg.gauss_sigma, cfg.gauss_tau)
        gt_neg = build_gaussian_gt(None, neg_id, cfg.localizers_path, cfg.gauss_sigma, cfg.gauss_tau)
        run_sliding_inference(cfg, pos_id, image_out / "prob_pos", gt_gauss=gt_pos)
        run_sliding_inference(cfg, neg_id, image_out / "prob_neg", gt_gauss=gt_neg)
    else:
        print("[Warn] Skipping inference examples (missing checkpoint or series IDs).")

    # Layer stats plotting: run a quick forward on a patch to populate activations
    try:
        sample_series = pos_id or neg_id
        if sample_series:
            with h5py.File(Path("train_dataset.h5"), "r") as f:
                vol = f["series"][sample_series]["vol"][:]
            vol = np.nan_to_num(np.asarray(vol, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            model = load_unet_from_config(base_output / "hyperparameters.json")
            state = torch.load(ckpt_candidates[0], map_location="cpu")
            model.load_state_dict(state, strict=False)
            model.to(torch.device(cfg.device))
            _, activations, gradients = get_layer_data(model, vol, patch_size=cfg.patch_size)
            plot_layer_stats(activations, gradients, output_folder / "visualizations")
        else:
            print("[Info] No sample series available for layer stats.")
    except Exception as e:
        print(f"[Warn] Could not generate layer stats: {e}")


if __name__ == "__main__":
    main()
