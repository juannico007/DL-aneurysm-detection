from torch.nn import utils
from .unet import UNet
from .data_augmentation import DataAugmentation, rotate_batch_gpu
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch.nn.functional as F
import bitsandbytes as bnb
from pathlib import Path
from accelerate import Accelerator
import pandas as pd
import csv
from dataclasses import dataclass
import h5py
import math

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        device = inputs.device
        inputs = inputs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32) 
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum(dtype=torch.float32)
        inputs_sum   = inputs.sum(dtype=torch.float32)
        targets_sum  = targets.sum(dtype=torch.float32)                        
        dice = (2.*intersection + smooth)/(inputs_sum.sum() + targets_sum.sum() + smooth)  
        return 1 - dice

class DiceSemimetricLoss(nn.Module):
    """
    JML2-style Dice semimetric loss that supports soft labels.
    For hard labels this matches classic soft Dice; for soft labels it keeps the optimum at x=y.
    """
    def __init__(self, eps: float = 1e-6, clip_threshold: float = 0.0, sharp_power: float = 1.0):
        super().__init__()
        self.eps = eps
        self.clip_threshold = clip_threshold
        self.sharp_power = sharp_power

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: logits, targets: soft labels
        probs = torch.sigmoid(inputs)
        probs = probs.to(dtype=torch.float32)
        targets = targets.to(dtype=torch.float32)
        # Sharpen the target to downweight tails of the Gaussian and focus on the center.
        targets = targets.pow(self.sharp_power)
        if self.clip_threshold > 0:
            targets = torch.where(targets >= self.clip_threshold, targets, torch.zeros_like(targets))

        # If we predict single-channel but have multi-heatmap targets, broadcast
        if probs.shape[1] == 1 and targets.shape[1] > 1:
            probs = probs.expand(-1, targets.shape[1], -1, -1, -1)

        intersection = (probs * targets).sum(dim=(2, 3, 4))
        sym_diff = (probs - targets).abs().sum(dim=(2, 3, 4))
        loss = 1.0 - (intersection + self.eps) / (intersection + sym_diff + self.eps)
        return loss.mean()
    
class SegmentationClassificationLoss(nn.Module):
    """
    Class for dice loss with bce for classification
    Parameters:
        seg_weight: weight for segmentation loss
        cls_weight: weight for classification loss
        segmentation: Function to evaluate the segmentation loss
        classification: Function to evaluate the classification loss
    Returns: 
        Loss value for the patient
    """
    def __init__(self, segmentation = DiceLoss, classification = nn.BCEWithLogitsLoss, seg_weight: float = 0.5, cls_weight: float = 0.5, neg_mean_lambda: float = 0.0, neg_max_lambda: float = 0.0, size_average=True):
        super(SegmentationClassificationLoss, self).__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        print("weights:", {"seg": self.seg_weight, "cls": self.cls_weight, "neg_mean": neg_mean_lambda, "neg_max": neg_max_lambda})
        self.segmentation = segmentation
        self.classification = classification
        self.neg_mean_lambda = float(neg_mean_lambda)
        self.neg_max_lambda = float(neg_max_lambda)

    def forward(self, inputs, targets, device):
        #Divide inputs and targets for dice loss and bce
        segmentation_input = inputs[0]
        classification_input = inputs[1]
        segmentation_target = targets[0]
        classification_target = targets[1]
        classification_input = classification_input.to(device=device, dtype=torch.float32) 
        classification_target = classification_target.to(device=device, dtype=torch.float32) 
        #print("IN LOSS - seg_in.dtype, device, requires_grad:", segmentation_input.dtype, segmentation_input.device, segmentation_input.requires_grad)
        #print("IN LOSS - cls_in.dtype, device, requires_grad:", classification_input.dtype, classification_input.device, classification_input.requires_grad)
        
        segmentation_loss = self._segmentation_loss(segmentation_input, segmentation_target)

        # Penalize segmentation probabilities on negative patients to suppress false positives
        if self.neg_mean_lambda > 0 or self.neg_max_lambda > 0:
            probs = torch.sigmoid(segmentation_input).to(dtype=torch.float32)
            neg_mask = (classification_target.squeeze(1) <= 0.5)
            if neg_mask.any():
                neg_probs = probs[neg_mask]
                if self.neg_mean_lambda > 0:
                    segmentation_loss = segmentation_loss + self.neg_mean_lambda * neg_probs.mean()
                if self.neg_max_lambda > 0:
                    segmentation_loss = segmentation_loss + self.neg_max_lambda * neg_probs.max()

        if self.cls_weight != 0:
            classification_loss = self.classification(classification_input, classification_target)
            return self.seg_weight * segmentation_loss + self.cls_weight * classification_loss
        return segmentation_loss

    def _segmentation_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute segmentation loss using the provided segmentation criterion (supports soft labels)."""
        return self.segmentation(inputs, targets)
    
def custom_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def _round_up(n: int, m: int = 16) -> int:
    return ((n + m - 1) // m) * m

def pad_collate_3d(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], multiple: int = 16):
    """
    Each x in batch has shape (C=1, D, H, W) and sizes may differ.
    We pad on the right to (Dt,Ht,Wt) = per-batch max (rounded to 'multiple'),
    then stack into (B, 1, Dt, Ht, Wt).
    """
    xs, ms, ys = zip(*batch)  # xs: tuple of tensors (1,D,H,W), ys: tuple of ints/tensors

    # ensure labels tensor (B,)
    Y = torch.as_tensor(ys, dtype=torch.long)

    # get per-batch target size
    shapes = [x.shape[-3:] for x in xs]  # (D,H,W)
    Dm, Hm, Wm = (max(s[i] for s in shapes) for i in range(3))
    Dt, Ht, Wt = _round_up(Dm, multiple), _round_up(Hm, multiple), _round_up(Wm, multiple)

    padded_x, padded_m = [], []
    for x, m in zip(xs, ms):
        # x is (1,D,H,W)
        _, D, H, W = x.shape
        pd, ph, pw = Dt - D, Ht - H, Wt - W
        # pad order: (W_left, W_right, H_left, H_right, D_left, D_right)
        x = F.pad(x, (0,pw,0,ph,0,pd), value=0.0)
        m = F.pad(m, (0,pw,0,ph,0,pd), value=0)
        padded_x.append(x)
        padded_m.append(m)

    X = torch.stack(padded_x, dim=0)
    M = torch.stack(padded_m, dim=0)
    return X, M, Y

def make_sphere_mask(shape, center, radius=5.0):
    """Return a binary 3D mask with a filled sphere."""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cy, cx = center
    dist = np.square(x - cx)+ np.square(y - cy) + np.square(z - cz)
    mask = dist <= radius**2
    return mask.astype(np.uint8)

def _gaussian_heatmap(shape: Tuple[int, int, int], center: Tuple[float, float, float], sigma: float) -> np.ndarray:
    """Create a single 3D Gaussian heatmap with peak 1.0."""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cy, cx = center
    dist2 = np.square(x - cx)+ np.square(y - cy) + np.square(z - cz)
    heatmap = np.exp(-dist2 / (2.0 * sigma ** 2))
    peak = heatmap.max()
    if peak > 0:
        heatmap = heatmap / peak
    return heatmap.astype(np.float32)


def make_heatmaps(shape: Tuple[int, int, int], center: Tuple[float, float, float], sigmas: Sequence[float]) -> np.ndarray:
    """Return stacked 3D Gaussian heatmaps for each sigma."""
    if not sigmas:
        return np.zeros((1,) + tuple(shape), dtype=np.float32)
    heatmaps = [_gaussian_heatmap(shape, center, sigma) for sigma in sigmas]
    return np.stack(heatmaps, axis=0)


@dataclass
class PatchRecord:
    series_id: str
    center_z: float
    center_y: float
    center_x: float
    label: int
    aneurysm_x: Optional[float] = None
    aneurysm_y: Optional[float] = None
    aneurysm_z: Optional[float] = None
    aneurysm_rel_x: Optional[float] = None
    aneurysm_rel_y: Optional[float] = None
    aneurysm_rel_z: Optional[float] = None
    location: str = "background"

class AneurysmDataset(Dataset):
    """Lightweight dataset that fetches volumes (and optional masks) from HDF5 on demand."""

    def __init__(
        self,
        h5_path: Path,
        series_ids: Sequence[str],
        labels: Sequence[int],
        transform=None,
        localizer_csv: Optional[Path] = None,
        radius: float = 5.0,
        heatmap_sizes: Sequence[float] = (15.0,),
    ):
        self.h5_path = str(h5_path)
        self.series_ids = list(series_ids)
        self.labels = [int(label) for label in labels]
        self.transform = transform
        self.radius = radius
        self.heatmap_sizes = list(heatmap_sizes) if heatmap_sizes else [radius]
        self.localizer_points = self._load_localizer_points(localizer_csv)
        self._h5 = None

    def __len__(self):
        return len(self.series_ids)

    def _get_file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self):
        self.close()

    # returns a dict mapping SeriesInstanceUID to np.ndarray of shape (N, 3) with (z_new,y_new,x_new) points
    def _load_localizer_points(self, csv_path: Optional[Path]) -> Dict[str, np.ndarray]:
        if not csv_path:
            return {}
        csv_path = Path(csv_path).expanduser()
        if not csv_path.exists():
            print(f"[Warning] Localizer CSV not found: {csv_path}. Masks disabled.")
            return {}
        df = pd.read_csv(csv_path)
        required = {"SeriesInstanceUID", "x_new", "y_new", "z_new"}
        if not required.issubset(df.columns):
            print(
                "[Warning] Localizer CSV missing required columns "
                f"({', '.join(sorted(required))}). Masks disabled."
            )
            return {}
        grouped = {}
        for uid, group in df.groupby("SeriesInstanceUID"):
            grouped[str(uid)] = group[["z_new", "y_new", "x_new"]].to_numpy(dtype=float)
        return grouped

    def _build_mask(self, pid: str, shape: Tuple[int, int, int]) -> torch.Tensor:
        mask_np = np.zeros((len(self.heatmap_sizes),) + shape, dtype=np.float32)
        centers = self.localizer_points.get(str(pid))
        if centers is not None:
            for center in centers:
                heatmaps = make_heatmaps(shape, center, sigmas=self.heatmap_sizes)
                mask_np = np.maximum(mask_np, heatmaps)
        return torch.from_numpy(mask_np)

    @staticmethod
    def _empty_mask(shape: Tuple[int, int, int], channels: int = 1) -> torch.Tensor:
        return torch.zeros((channels,) + tuple(shape), dtype=torch.float32)

    def __getitem__(self, idx):
        handle = self._get_file()
        pid = self.series_ids[idx]
        group = handle["series"][pid]
        volume = torch.from_numpy(group["vol"][:])
        centers = self.localizer_points.get(str(pid))
        if centers is not None and len(centers):
            mask = self._build_mask(pid, volume.shape[-3:])
        else:
            mask = self._empty_mask(volume.shape[-3:], channels=len(self.heatmap_sizes))

        if self.transform:
            volume = self.transform(volume)
        elif volume.ndim == 3:
            volume = volume.unsqueeze(0)

        label = self.labels[idx]
        return volume, mask, label


def _safe_float(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_patch_records(patch_csv: Path) -> List[PatchRecord]:
    """Load patch metadata from CSV into PatchRecord objects."""
    records: List[PatchRecord] = []
    with patch_csv.open() as f:
        reader = csv.DictReader(f)
        required = {"series_id", "center_x", "center_y", "center_z", "label"}
        missing = required.difference(reader.fieldnames or set())
        if missing:
            raise ValueError(f"Patches CSV missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            label = int(row["label"])
            records.append(
                PatchRecord(
                    series_id=row["series_id"],
                    center_z=float(row["center_z"]),
                    center_y=float(row["center_y"]),
                    center_x=float(row["center_x"]),
                    label=label,
                    aneurysm_x=_safe_float(row.get("aneurysm_x")),
                    aneurysm_y=_safe_float(row.get("aneurysm_y")),
                    aneurysm_z=_safe_float(row.get("aneurysm_z")),
                    aneurysm_rel_x=_safe_float(row.get("aneurysm_rel_x")),
                    aneurysm_rel_y=_safe_float(row.get("aneurysm_rel_y")),
                    aneurysm_rel_z=_safe_float(row.get("aneurysm_rel_z")),
                    location=row.get("location", "background") or "background",
                )
            )
    return records


class PatchAneurysmDataset(Dataset):
    """Dataset that fetches fixed-size patches from HDF5 using patches.csv metadata."""

    def __init__(
        self,
        h5_path: Path,
        records: List[PatchRecord],
        patch_size: int,
        transform=None,
        radius: float = 5.0,
        heatmap_sizes: Sequence[float] = (15.0,),
    ):
        self.h5_path = str(h5_path)
        self.records = records
        self.patch_size = patch_size
        self.transform = transform
        self.radius = radius
        self.heatmap_sizes = list(heatmap_sizes) if heatmap_sizes else [radius]
        self._h5 = None

    def __len__(self):
        return len(self.records)

    def _get_file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self):
        self.close()

    def _crop_patch(self, vol_ds, center: Tuple[float, float, float]) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        half = self.patch_size / 2.0
        starts = [
            int(max(0, min(vol_ds.shape[i] - self.patch_size, round(center[i] - half))))
            for i in range(3)
        ]
        z0, y0, x0 = starts
        z1, y1, x1 = z0 + self.patch_size, y0 + self.patch_size, x0 + self.patch_size
        patch = vol_ds[z0:z1, y0:y1, x0:x1]
        return patch, (z0, y0, x0)

    def __getitem__(self, idx):
        record = self.records[idx]
        handle = self._get_file()
        vol_ds = handle["series"][record.series_id]["vol"]
        patch_np, (z0, y0, x0) = self._crop_patch(vol_ds, (record.center_z, record.center_y, record.center_x))

        if patch_np.shape != (self.patch_size, self.patch_size, self.patch_size):
            # pad if at boundary
            pad_z = self.patch_size - patch_np.shape[0]
            pad_y = self.patch_size - patch_np.shape[1]
            pad_x = self.patch_size - patch_np.shape[2]
            patch_np = np.pad(patch_np, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")

        mask_np = np.zeros((len(self.heatmap_sizes),) + patch_np.shape, dtype=np.float32)
        if record.label == 1:
            if record.aneurysm_rel_x is not None and record.aneurysm_rel_y is not None and record.aneurysm_rel_z is not None:
                center = (record.aneurysm_rel_z, record.aneurysm_rel_y, record.aneurysm_rel_x)
            elif record.aneurysm_x is not None and record.aneurysm_y is not None and record.aneurysm_z is not None:
                center = (
                    record.aneurysm_z - z0,
                    record.aneurysm_y - y0,
                    record.aneurysm_x - x0,
                )
            else:
                center = (self.patch_size / 2.0, self.patch_size / 2.0, self.patch_size / 2.0)
            mask_np = make_heatmaps(patch_np.shape, center, sigmas=self.heatmap_sizes)

        volume = torch.from_numpy(patch_np)
        mask = torch.from_numpy(mask_np)

        if self.transform:
            volume = self.transform(volume)
        elif volume.ndim == 3:
            volume = volume.unsqueeze(0)

        return volume, mask, record.label

class TrainingPipeline:
    """Training pipeline for the aneurysm detection model."""
    
    def __init__(
        self,
        model: UNet,
        batch_size: int = 4,
        epochs: int = 4,
        learning_rate : float = 1e-4,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 20,
        lr_reduction_patience: int = 10,
        lr_reduction_factor : float = 0.5,
        min_lr : float = 1e-7,
        checkpoint_path: str = "aneurysm_detection_best.pt",
        mixed_precision: str = "fp16",
        grad_accum_steps: int = 2,
        radius: float = 5.0,
        heatmap_sigma: float = 15.0,
        heatmap_decay_epoch: int = 0,
        patch_csv: Optional[Path] = None,
        patch_size: int = 64,
        train_ratio: float = 0.8,
        split_seed: int = 42,
        scheduler_config: Optional[Dict[str, Any]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        neg_warmup_epochs: int = 0,
        ):
        """
        Initialize training pipeline.
        
        Parameters
        ----------
            model: AneurysmDetectionModel instance
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Initial learning rate for the model
            early_stopping_patience: Number of epochs of no improving until early stopping
            lr_reduction_patience: Number of epochs of plateau until reducing lr
            lr_reduction_factor: Learning rate reduction on plateau
            min_lr: Minimum value for the learning rate
            checkpoint_path: Path to save best model
            loss_weights: dict with weights for segmentation/classification and negative penalties
        """
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.lr_reduction_patience = lr_reduction_patience
        self.lr_reduction_factor = lr_reduction_factor
        self.min_lr = min_lr
        self.checkpoint_path = checkpoint_path
        self.radius = radius
        self.initial_heatmap_sigma = float(heatmap_sigma)
        self.current_heatmap_sigma = float(heatmap_sigma)
        self.heatmap_min_sigma = 5.0
        self.heatmap_decay_epoch = int(heatmap_decay_epoch)
        self.heatmap_sizes = [self.current_heatmap_sigma]
        self.primary_heatmap_index = 0
        self.patch_csv = Path(patch_csv) if patch_csv else None
        self.patch_size = patch_size
        self.train_ratio = train_ratio
        self.split_seed = split_seed
        self.scheduler_config = dict(scheduler_config) if scheduler_config else {
            "name": "step",
            "step_size": 100,
            "gamma": 0.96,
        }
        lw = loss_weights or {}
        self.seg_weight = float(lw.get("segmentation", 0.7))
        self.cls_weight = float(lw.get("classification", 0.3))
        self.base_neg_mean_lambda = float(lw.get("neg_mean", 0.1))
        self.base_neg_max_lambda = float(lw.get("neg_max", 0.05))
        self.neg_mean_lambda = self.base_neg_mean_lambda
        self.neg_max_lambda = self.base_neg_max_lambda
        self.neg_warmup_epochs = int(neg_warmup_epochs)
        self.neg_ramp_epochs = max(1, self.epochs - self.neg_warmup_epochs)

        self.scheduler_step_mode: Optional[str] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.scheduler_name: Optional[str] = None

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision, 
            gradient_accumulation_steps=grad_accum_steps
        )

    def _set_heatmap_sigma(self, sigma: float):
        """Update heatmap sigma (clamped to minimum) across datasets."""
        self.current_heatmap_sigma = max(self.heatmap_min_sigma, float(sigma))
        self.heatmap_sizes = [self.current_heatmap_sigma]
        if hasattr(self, "train_loader") and hasattr(self.train_loader, "dataset"):
            self.train_loader.dataset.heatmap_sizes = self.heatmap_sizes
        if hasattr(self, "val_loader") and hasattr(self.val_loader, "dataset"):
            self.val_loader.dataset.heatmap_sizes = self.heatmap_sizes

    def _sigma_for_epoch(self, epoch: int) -> float:
        """Compute decayed sigma rounded to int, reducing by 1 every `heatmap_decay_epoch` epochs."""
        interval = self.heatmap_decay_epoch
        if not interval or interval <= 0:
            return self.initial_heatmap_sigma
        decays = epoch // interval
        decayed = round(self.initial_heatmap_sigma - decays)
        return max(self.heatmap_min_sigma, decayed)

    def _maybe_decay_heatmap(self, epoch: int):
        """Shrink sigma gradually after the configured epoch, rounded to int."""
        new_sigma = self._sigma_for_epoch(epoch)
        if new_sigma != self.current_heatmap_sigma:
            self.accelerator.print(f"[Heatmap] Sigma update at epoch {epoch}: {self.current_heatmap_sigma} -> {new_sigma}")
        self._set_heatmap_sigma(new_sigma)

    def _neg_penalty_scale(self, epoch: int) -> float:
        """Gaussian ramp from 0 to 1 after warmup for negative suppression terms."""
        if self.neg_ramp_epochs <= 0:
            return 1.0
        t = (epoch - self.neg_warmup_epochs) / max(1, self.neg_ramp_epochs)
        t = min(max(t, 0.0), 1.0)
        return math.exp(-5.0 * (1.0 - t) * (1.0 - t))

    def _update_neg_lambdas(self, epoch: int):
        """Update neg penalty lambdas according to ramp schedule."""
        scale = self._neg_penalty_scale(epoch)
        self.neg_mean_lambda = self.base_neg_mean_lambda * scale
        self.neg_max_lambda = self.base_neg_max_lambda * scale
        if hasattr(self, "criterion"):
            self.criterion.neg_mean_lambda = self.neg_mean_lambda
            self.criterion.neg_max_lambda = self.neg_max_lambda

    def create_dataloaders(
        self,
        h5_path: Path,
        patch_csv: Path,
        train_ratio: Optional[float] = None,
        split_seed: Optional[int] = None,
        records: Optional[List[PatchRecord]] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders from patch metadata.
        """
        if patch_csv is None and records is None and self.patch_csv is None:
            raise ValueError("patch_csv or records must be provided for patch-based training.")
        patch_csv = patch_csv or self.patch_csv
        train_ratio = train_ratio if train_ratio is not None else self.train_ratio
        split_seed = split_seed if split_seed is not None else self.split_seed

        if records is None:
            records = load_patch_records(Path(patch_csv))
        if not records:
            raise ValueError(f"No patch records found in {patch_csv}")

        rng = np.random.default_rng(split_seed)
        indices = np.arange(len(records))
        rng.shuffle(indices)
        cutoff = int(train_ratio * len(indices))
        train_idx = indices[:cutoff]
        val_idx = indices[cutoff:]

        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]

        train_labels = np.array([r.label for r in train_records], dtype=int)
        pos = train_labels.sum()
        neg = len(train_labels) - pos
        if pos == 0:
            pos_weight = torch.tensor(1.0)
        else:
            pos_weight = torch.tensor(neg / max(pos, 1), dtype=torch.float32)
        self.pos_weight = pos_weight

        train_transforms = transforms.Compose([
            DataAugmentation.augment_training
        ])
        val_transforms = transforms.Compose([
            DataAugmentation.prepare_validation
        ])

        train_dataset = PatchAneurysmDataset(
            h5_path=h5_path,
            records=train_records,
            patch_size=self.patch_size,
            transform=train_transforms,
            radius=self.radius,
            heatmap_sizes=self.heatmap_sizes,
        )
        val_dataset = PatchAneurysmDataset(
            h5_path=h5_path,
            records=val_records,
            patch_size=self.patch_size,
            transform=val_transforms,
            radius=self.radius,
            heatmap_sizes=self.heatmap_sizes,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=pad_collate_3d,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=pad_collate_3d,
        )
        self.train_loader, self.val_loader = self.accelerator.prepare(train_loader, val_loader)
        return self.train_loader, self.val_loader

    def _configure_scheduler(self):
        """Create the requested LR scheduler and remember how it should be stepped."""
        config = dict(self.scheduler_config) if self.scheduler_config else {}
        name = config.get("name", "step")
        if not name:
            self.scheduler = None
            self.scheduler_step_mode = None
            self.scheduler_name = None
            return

        name = name.lower()
        self.scheduler_name = name
        if name in {"none", "off"}:
            self.scheduler = None
            self.scheduler_step_mode = None
            return

        if name == "step":
            step_size = int(config.get("step_size", 100))
            gamma = float(config.get("gamma", 0.96))
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma,
            )
            self.scheduler_step_mode = "epoch"
        elif name == "cosine":
            t_max = int(config.get("T_max", self.epochs))
            eta_min = float(config.get("eta_min", 1e-6))
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )
            self.scheduler_step_mode = "epoch"
        elif name in {"plateau", "reduce_on_plateau"}:
            factor = float(config.get("factor", 0.5))
            patience = int(config.get("patience", 10))
            min_lr = float(config.get("min_lr", 1e-6))
            mode = config.get("mode", "min")
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )
            self.scheduler_step_mode = "plateau"
        elif name in {"onecycle", "one_cycle"}:
            if not hasattr(self, "train_loader"):
                raise ValueError("OneCycleLR scheduler requires dataloaders to be created before training.")
            steps_per_epoch = int(config.get("steps_per_epoch", len(self.train_loader)))
            epochs = int(config.get("epochs", self.epochs))
            max_lr = float(config.get("max_lr", self.learning_rate))
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
            )
            self.scheduler_step_mode = "batch"
        else:
            raise ValueError(f"Unsupported scheduler '{name}'.")
    
    def _safe_gather(self, tensor):
        """Safely gather tensors across processes for Gloo backend."""
        tensor = tensor.flatten().contiguous()
        
        # Get max size across processes
        local_size = torch.tensor(tensor.shape[0], device=self.device)
        all_sizes = self.accelerator.gather(local_size)
        if not torch.is_tensor(all_sizes):
            all_sizes = torch.tensor([all_sizes], device=self.device)
        else:
            all_sizes = all_sizes.to(self.device)
        max_size = all_sizes.max().item() if all_sizes.numel() else 0
        
        # Pad to max size
        current_size = tensor.shape[0]
        if current_size < max_size:
            tensor = torch.nn.functional.pad(tensor, (0, max_size - current_size), value=0)

        gathered = self.accelerator.gather(tensor)

        if max_size == 0:
            return gathered

        sizes_list = all_sizes.tolist()
        if isinstance(sizes_list, (int, float)):
            sizes_list = [sizes_list]
        masks = [
            (torch.arange(max_size, device=self.device) < size)
            for size in sizes_list
        ]
        valid_mask = torch.stack(masks, dim=0).flatten()
        return gathered[valid_mask]
    
    def train(
        self,
    ) -> dict:
        """Run the training loop using the loaders prepared beforehand."""
        # # Check if GPU available
        # if torch.cuda.is_available():
        #     print("GPU is on")
        #     self.device = "cuda"
        #     self.model = self.model.to("cuda")

        self.device = self.accelerator.device

        self.optimizer = bnb.optim.Adam8bit(
            self.model.parameters(), 
            lr=self.learning_rate,
            betas = (0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay)
        self.segmentation_loss = DiceSemimetricLoss()
        cls_pos_weight = getattr(self, "pos_weight", torch.tensor(1.0))
        cls_pos_weight = cls_pos_weight.to(self.device)
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight)
        self.criterion = SegmentationClassificationLoss(
            self.segmentation_loss,
            self.classification_loss,
            seg_weight=self.seg_weight,
            cls_weight=self.cls_weight,
            neg_mean_lambda=self.neg_mean_lambda,
            neg_max_lambda=self.neg_max_lambda,
        )
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self._configure_scheduler()
        #Since we have to define our own training loop, we have to keep track
        #of these variables for best model, early stopping and learning rate decrease
        best_val_loss = 1
        epochs_no_improve = 0
        lr_plateau_counter = 0

        #Also we want to keep track of our training history
        history = {
            "train_loss": [], "train_acc": [], "train_sensitivity": [], "train_ppv": [], "train_npv": [],
            "val_loss": [], "val_acc": [], "val_sensitivity": [], "val_ppv": [], "val_npv": [],
            "train_dice": [], "val_dice": []
        }

        eps = 1e-6
        for epoch in range(1, self.epochs + 1):
            self._maybe_decay_heatmap(epoch)
            self._update_neg_lambdas(epoch)
            self.model.train()
            train_losses = []
            train_dice_scores = []
            train_core_list, train_peak_list, train_focus_list = [], [], []
            all_preds, all_labels = [], []
            train_neg_mean_list, train_neg_max_list = [], []

            for x_batch, mask_batch, y_batch in tqdm(self.train_loader):
                x_batch = x_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                mask_batch = mask_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                y_batch = y_batch.float().unsqueeze(1)
                y_batch = y_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                x_batch, angles = rotate_batch_gpu(x_batch, mode='bilinear')
                mask_batch, _ = rotate_batch_gpu(mask_batch, angles=angles, mode='nearest')
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        outputs = self.model(x_batch)
                    #print("Classifier logits:", outputs[1].min(), "true label:", y_batch)
                    loss = self.criterion(outputs, (mask_batch, y_batch), self.device)
                    # just before accelerator.backward(loss)
                    #print("LOSS:", loss, "device:", loss.device, "dtype:", loss.dtype, "requires_grad:", loss.requires_grad)
                    seg_logits, cls_logits = outputs
                    #print("seg_logits.requires_grad:", seg_logits.requires_grad, "device:", seg_logits.device, "dtype:", seg_logits.dtype)
                    #print("cls_logits.requires_grad:", cls_logits.requires_grad, "device:", cls_logits.device, "dtype:", cls_logits.dtype)
                    with torch.autograd.set_detect_anomaly(True):
                        self.accelerator.backward(loss)

                    # IMPORTANT — only clip when gradients exist
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler and self.scheduler_step_mode == "batch":
                        self.scheduler.step()

                train_losses.append(self.accelerator.gather(loss.detach()).mean().item())

                #Dice calculation (soft dice on soft targets)
                with torch.no_grad():
                    mask_probs = torch.sigmoid(outputs[0].detach().to(torch.float32))
                    targets = mask_batch[:, self.primary_heatmap_index:self.primary_heatmap_index+1].detach().to(torch.float32)
                    pos_mask = (y_batch.squeeze(1) > 0.5)
                    if pos_mask.any():
                        mask_probs_pos = mask_probs[pos_mask]
                        targets_pos = targets[pos_mask]
                    else:
                        mask_probs_pos = targets_pos = None
                    neg_mask = (y_batch.squeeze(1) <= 0.5)
                    if neg_mask.any():
                        mask_probs_neg = mask_probs[neg_mask]
                        neg_mean = mask_probs_neg.mean(dim=(1, 2, 3, 4))
                        neg_max = mask_probs_neg.amax(dim=(1, 2, 3, 4))
                        train_neg_mean_list.append(neg_mean)
                        train_neg_max_list.append(neg_max)

                    pred_sum = mask_probs.sum(dim=(1, 2, 3, 4))
                    target_sum = targets.sum(dim=(1, 2, 3, 4))
                    intersection = (mask_probs * targets).sum(dim=(1, 2, 3, 4))
                    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
                    train_dice_scores.append(dice)

                    # Binary core dice: threshold target and prediction
                    if targets_pos is not None:
                        tgt_bin = (targets_pos >= 0.1).float()
                        pred_bin = (mask_probs_pos >= 0.5).float()
                        pred_bsum = pred_bin.sum(dim=(1, 2, 3, 4))
                        tgt_bsum = tgt_bin.sum(dim=(1, 2, 3, 4))
                        inter_bin = (pred_bin * tgt_bin).sum(dim=(1, 2, 3, 4))
                        core_dice = (2.0 * inter_bin + eps) / (pred_bsum + tgt_bsum + eps)

                        # Peak localization error (voxels)
                        pred_peak = mask_probs_pos.view(mask_probs_pos.shape[0], -1).argmax(dim=1)
                        tgt_peak = targets_pos.view(targets_pos.shape[0], -1).argmax(dim=1)
                        # convert flat index to z,y,x
                        def unravel(idx, shape):
                            d, h, w = shape
                            z = idx // (h * w)
                            y = (idx % (h * w)) // w
                            x = idx % w
                            return z, y, x
                        shape = targets_pos.shape[-3:]
                        pred_peak_coords = torch.stack([torch.tensor(unravel(i.item(), shape), device=mask_probs.device) for i in pred_peak])
                        tgt_peak_coords = torch.stack([torch.tensor(unravel(i.item(), shape), device=mask_probs.device) for i in tgt_peak])
                        peak_err = torch.norm(pred_peak_coords.to(torch.float32) - tgt_peak_coords.to(torch.float32), dim=1)

                        # Focus ratio: mass inside radius vs total
                        radius = 5.0
                        grid_z = torch.arange(shape[0], device=mask_probs.device).view(-1, 1, 1)
                        grid_y = torch.arange(shape[1], device=mask_probs.device).view(1, -1, 1)
                        grid_x = torch.arange(shape[2], device=mask_probs.device).view(1, 1, -1)
                        focus_scores = []
                        for b in range(mask_probs_pos.shape[0]):
                            cz, cy, cx = tgt_peak_coords[b]
                            dist2 = (grid_z - cz) ** 2 + (grid_y - cy) ** 2 + (grid_x - cx) ** 2
                            in_mask = dist2 <= radius ** 2
                            mass_total = mask_probs_pos[b, 0].sum()
                            mass_focus = (mask_probs_pos[b, 0] * in_mask).sum()
                            focus_scores.append((mass_focus / (mass_total + eps)))
                        focus_scores = torch.stack(focus_scores)
                    else:
                        core_dice = torch.tensor([], device=self.device)
                        peak_err = torch.tensor([], device=self.device)
                        focus_scores = torch.tensor([], device=self.device)

                #Aggregate metrics
                all_preds.append(outputs[1].detach())
                all_labels.append(y_batch.detach())
                train_core_list.append(core_dice)
                train_peak_list.append(peak_err)
                train_focus_list.append(focus_scores)

            #DICE loss stats
            train_dice_scores = torch.cat(train_dice_scores, dim=0).to(self.device, dtype=torch.float32)
            train_dice_scores = self._safe_gather(train_dice_scores)
            train_dice_mean = train_dice_scores.cpu().numpy().mean() if train_dice_scores.numel() > 0 else 0.0
            # Core dice / peak / focus stats
            core_all = torch.cat(train_core_list, dim=0).to(self.device, dtype=torch.float32) if train_core_list else torch.tensor([], device=self.device)
            peak_all = torch.cat(train_peak_list, dim=0).to(self.device, dtype=torch.float32) if train_peak_list else torch.tensor([], device=self.device)
            focus_all = torch.cat(train_focus_list, dim=0).to(self.device, dtype=torch.float32) if train_focus_list else torch.tensor([], device=self.device)
            core_all = self._safe_gather(core_all)
            peak_all = self._safe_gather(peak_all)
            focus_all = self._safe_gather(focus_all)
            train_core_dice_mean = core_all.cpu().numpy().mean() if core_all.numel() > 0 else 0.0
            train_peak_mean = peak_all.cpu().numpy().mean() if peak_all.numel() > 0 else 0.0
            train_focus_mean = focus_all.cpu().numpy().mean() if focus_all.numel() > 0 else 0.0
            # Negative-only segmentation stats
            neg_mean_all = torch.cat(train_neg_mean_list, dim=0).to(self.device, dtype=torch.float32) if train_neg_mean_list else torch.tensor([], device=self.device)
            neg_max_all = torch.cat(train_neg_max_list, dim=0).to(self.device, dtype=torch.float32) if train_neg_max_list else torch.tensor([], device=self.device)
            neg_mean_all = self._safe_gather(neg_mean_all)
            neg_max_all = self._safe_gather(neg_max_all)
            train_neg_prob_mean = neg_mean_all.cpu().numpy().mean() if neg_mean_all.numel() > 0 else 0.0
            train_neg_prob_max = neg_max_all.cpu().numpy().mean() if neg_max_all.numel() > 0 else 0.0

            #Classification stats
            all_preds = torch.cat(all_preds, dim=0).to(self.device, dtype=torch.float32).contiguous()
            all_labels = torch.cat(all_labels, dim=0).to(self.device, dtype=torch.float32).contiguous()
            all_preds  = self._safe_gather(all_preds)
            all_labels = self._safe_gather(all_labels)

            probs = torch.sigmoid(all_preds)
            train_preds_np = (probs >= 0.5).int().cpu().numpy().ravel()
            train_labels_np  = all_labels.int().cpu().numpy().ravel()

            tp = np.logical_and(train_preds_np == 1, train_labels_np == 1).sum()
            tn = np.logical_and(train_preds_np == 0, train_labels_np == 0).sum()
            fp = np.logical_and(train_preds_np == 1, train_labels_np == 0).sum()
            fn = np.logical_and(train_preds_np == 0, train_labels_np == 1).sum()
            train_acc = (tp + tn) / max(len(train_labels_np), 1)
            train_sens = tp / max(tp + fn, 1)  # sensitivity/recall
            train_ppv = tp / max(tp + fp, 1)   # precision/PPV
            train_npv = tn / max(tn + fn, 1)   # NPV

            history["train_loss"].append(np.mean(train_losses))
            history["train_acc"].append(train_acc)
            history["train_sensitivity"].append(train_sens)
            history["train_ppv"].append(train_ppv)
            history["train_npv"].append(train_npv)
            history["train_dice"].append(train_dice_mean)
            history.setdefault("train_core_dice_series", []).append(train_core_dice_mean)
            history.setdefault("train_peak_err_series", []).append(train_peak_mean)
            history.setdefault("train_focus_series", []).append(train_focus_mean)
            history.setdefault("train_neg_prob_mean_series", []).append(train_neg_prob_mean)
            history.setdefault("train_neg_prob_max_series", []).append(train_neg_prob_max)

            # --- Validation ---
            self.model.eval()
            val_losses = []
            val_dice_scores = []
            val_preds_list, val_labels_list = [], []
            with torch.no_grad():
                val_core_list, val_peak_list, val_focus_list = [], [], []
                val_neg_mean_list, val_neg_max_list = [], []
                for x_batch, mask_batch, y_batch in self.val_loader:
                    x_batch = x_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                    mask_batch = mask_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                    y_batch = y_batch.float().unsqueeze(1)
                    y_batch = y_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                    with self.accelerator.autocast():
                        outputs = self.model(x_batch)
                        loss = self.criterion(outputs, (mask_batch, y_batch), self.device)
                    val_losses.append(self.accelerator.gather(loss.detach()).mean().item())

                    #DICE loss (soft dice on soft targets)
                    mask_probs = torch.sigmoid(outputs[0].detach().to(torch.float32))
                    targets = mask_batch[:, self.primary_heatmap_index:self.primary_heatmap_index+1].detach().to(torch.float32)
                    pos_mask = (y_batch.squeeze(1) > 0.5)
                    if pos_mask.any():
                        mask_probs_pos = mask_probs[pos_mask]
                        targets_pos = targets[pos_mask]
                    else:
                        mask_probs_pos = targets_pos = None

                    pred_sum = mask_probs.sum(dim=(1, 2, 3, 4))
                    target_sum = targets.sum(dim=(1, 2, 3, 4))
                    intersection = (mask_probs * targets).sum(dim=(1, 2, 3, 4))
                    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
                    val_dice_scores.append(dice)

                    # Auxiliary metrics
                    if targets_pos is not None:
                        tgt_bin = (targets_pos >= 0.1).float()
                        pred_bin = (mask_probs_pos >= 0.5).float()
                        pred_bsum = pred_bin.sum(dim=(1, 2, 3, 4))
                        tgt_bsum = tgt_bin.sum(dim=(1, 2, 3, 4))
                        inter_bin = (pred_bin * tgt_bin).sum(dim=(1, 2, 3, 4))
                        core_dice = (2.0 * inter_bin + eps) / (pred_bsum + tgt_bsum + eps)

                        pred_peak = mask_probs_pos.view(mask_probs_pos.shape[0], -1).argmax(dim=1)
                        tgt_peak = targets_pos.view(targets_pos.shape[0], -1).argmax(dim=1)
                        shape = targets_pos.shape[-3:]
                        def unravel(idx, shape):
                            d, h, w = shape
                            z = idx // (h * w)
                            y = (idx % (h * w)) // w
                            x = idx % w
                            return z, y, x
                        pred_peak_coords = torch.stack([torch.tensor(unravel(i.item(), shape), device=mask_probs.device) for i in pred_peak])
                        tgt_peak_coords = torch.stack([torch.tensor(unravel(i.item(), shape), device=mask_probs.device) for i in tgt_peak])
                        peak_err = torch.norm(pred_peak_coords.to(torch.float32) - tgt_peak_coords.to(torch.float32), dim=1)

                        radius = 5.0
                        grid_z = torch.arange(shape[0], device=mask_probs.device).view(-1, 1, 1)
                        grid_y = torch.arange(shape[1], device=mask_probs.device).view(1, -1, 1)
                        grid_x = torch.arange(shape[2], device=mask_probs.device).view(1, 1, -1)
                        focus_scores = []
                        for b in range(mask_probs_pos.shape[0]):
                            cz, cy, cx = tgt_peak_coords[b]
                            dist2 = (grid_z - cz) ** 2 + (grid_y - cy) ** 2 + (grid_x - cx) ** 2
                            in_mask = dist2 <= radius ** 2
                            mass_total = mask_probs_pos[b, 0].sum()
                            mass_focus = (mask_probs_pos[b, 0] * in_mask).sum()
                            focus_scores.append((mass_focus / (mass_total + eps)))
                        focus_scores = torch.stack(focus_scores)
                    else:
                        core_dice = torch.tensor([], device=self.device)
                        peak_err = torch.tensor([], device=self.device)
                        focus_scores = torch.tensor([], device=self.device)
                    # Negative stats
                    neg_mask = (y_batch.squeeze(1) <= 0.5)
                    if neg_mask.any():
                        mask_probs_neg = mask_probs[neg_mask]
                        val_neg_mean_list.append(mask_probs_neg.mean(dim=(1, 2, 3, 4)))
                        val_neg_max_list.append(mask_probs_neg.amax(dim=(1, 2, 3, 4)))

                    val_core_list.append(core_dice)
                    val_peak_list.append(peak_err)
                    val_focus_list.append(focus_scores)

                    #Classification loss
                    preds = self._safe_gather(outputs[1].detach().to(torch.float32))
                    labels = self._safe_gather(y_batch.detach().to(torch.float32))

                    # Sigmoid + threshold (done in float32 for numerical stability)
                    probs = torch.sigmoid(preds)
                    preds_np = (probs >= 0.5).int().cpu().numpy()
                    labs_np  = labels.int().cpu().numpy()
                    val_preds_list.append(preds_np)
                    val_labels_list.append(labs_np)

                    del preds, labels, probs, preds_np, labs_np, outputs, loss
                    torch.cuda.empty_cache()

            val_dice_scores = torch.cat(val_dice_scores, dim=0).to(self.device, dtype=torch.float32)
            val_dice_scores = self._safe_gather(val_dice_scores)
            val_dice_mean = val_dice_scores.cpu().numpy().mean() if val_dice_scores.numel() > 0 else 0.0
            # Validation auxiliary metrics
            core_all = torch.cat(val_core_list, dim=0).to(self.device, dtype=torch.float32) if val_core_list else torch.tensor([], device=self.device)
            peak_all = torch.cat(val_peak_list, dim=0).to(self.device, dtype=torch.float32) if val_peak_list else torch.tensor([], device=self.device)
            focus_all = torch.cat(val_focus_list, dim=0).to(self.device, dtype=torch.float32) if val_focus_list else torch.tensor([], device=self.device)
            core_all = self._safe_gather(core_all)
            peak_all = self._safe_gather(peak_all)
            focus_all = self._safe_gather(focus_all)
            val_core_dice_mean = core_all.cpu().numpy().mean() if core_all.numel() > 0 else 0.0
            val_peak_mean = peak_all.cpu().numpy().mean() if peak_all.numel() > 0 else 0.0
            val_focus_mean = focus_all.cpu().numpy().mean() if focus_all.numel() > 0 else 0.0
            neg_mean_all = torch.cat(val_neg_mean_list, dim=0).to(self.device, dtype=torch.float32) if val_neg_mean_list else torch.tensor([], device=self.device)
            neg_max_all = torch.cat(val_neg_max_list, dim=0).to(self.device, dtype=torch.float32) if val_neg_max_list else torch.tensor([], device=self.device)
            neg_mean_all = self._safe_gather(neg_mean_all)
            neg_max_all = self._safe_gather(neg_max_all)
            val_neg_prob_mean = neg_mean_all.cpu().numpy().mean() if neg_mean_all.numel() > 0 else 0.0
            val_neg_prob_max = neg_max_all.cpu().numpy().mean() if neg_max_all.numel() > 0 else 0.0

            val_preds_np = np.concatenate(val_preds_list, axis=0).ravel()
            val_labels_np = np.concatenate(val_labels_list, axis=0).ravel()

            tp = np.logical_and(val_preds_np == 1, val_labels_np == 1).sum()
            tn = np.logical_and(val_preds_np == 0, val_labels_np == 0).sum()
            fp = np.logical_and(val_preds_np == 1, val_labels_np == 0).sum()
            fn = np.logical_and(val_preds_np == 0, val_labels_np == 1).sum()
            val_acc = (tp + tn) / max(len(val_labels_np), 1)
            val_sens = tp / max(tp + fn, 1)
            val_ppv = tp / max(tp + fp, 1)
            val_npv = tn / max(tn + fn, 1)
            val_loss = np.mean(val_losses)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_sensitivity"].append(val_sens)
            history["val_ppv"].append(val_ppv)
            history["val_npv"].append(val_npv)
            history["val_dice"].append(val_dice_mean)
            history.setdefault("val_core_dice_series", []).append(val_core_dice_mean)
            history.setdefault("val_peak_err_series", []).append(val_peak_mean)
            history.setdefault("val_focus_series", []).append(val_focus_mean)
            history.setdefault("val_neg_prob_mean_series", []).append(val_neg_prob_mean)
            history.setdefault("val_neg_prob_max_series", []).append(val_neg_prob_max)

            self.accelerator.print(
                f"Epoch {epoch}/{self.epochs}\n"
                f"  Train | Loss {history['train_loss'][-1]:.4f} | Acc {train_acc:.4f} | Sens {train_sens:.4f} | "
                f"PPV {train_ppv:.4f} | NPV {train_npv:.4f} | Dice {history['train_dice'][-1]:.4f} | "
                f"Core {train_core_dice_mean:.4f} | Peak {train_peak_mean:.2f} | Focus {train_focus_mean:.4f} | "
                f"NegMean {train_neg_prob_mean:.4f} | NegMax {train_neg_prob_max:.4f}\n"
                f"    Val | Loss {history['val_loss'][-1]:.4f} | Acc {val_acc:.4f} | Sens {val_sens:.4f} | "
                f"PPV {val_ppv:.4f} | NPV {val_npv:.4f} | Dice {history['val_dice'][-1]:.4f} | "
                f"Core {val_core_dice_mean:.4f} | Peak {val_peak_mean:.2f} | Focus {val_focus_mean:.4f} | "
                f"NegMean {val_neg_prob_mean:.4f} | NegMax {val_neg_prob_max:.4f}"
            )

            epoch_metrics = {
                "train/loss": history["train_loss"][-1],
                "train/acc": train_acc,
                "train/sensitivity": train_sens,
                "train/ppv": train_ppv,
                "train/npv": train_npv,
                "train/dice": history["train_dice"][-1],
                "train/core_dice": train_core_dice_mean,
                "train/peak_err": train_peak_mean,
                "train/focus": train_focus_mean,
                "train/neg_prob_mean": train_neg_prob_mean,
                "train/neg_prob_max": train_neg_prob_max,
                "val/loss": history["val_loss"][-1],
                "val/acc": val_acc,
                "val/sensitivity": val_sens,
                "val/ppv": val_ppv,
                "val/npv": val_npv,
                "val/dice": history["val_dice"][-1],
                "val/core_dice": val_core_dice_mean,
                "val/peak_err": val_peak_mean,
                "val/focus": val_focus_mean,
                "val/neg_prob_mean": val_neg_prob_mean,
                "val/neg_prob_max": val_neg_prob_max,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
            self.accelerator.log(epoch_metrics, step=epoch)

            # --- Checkpointing ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # unwrap for vanilla state_dict saving
                unwrapped = self.accelerator.unwrap_model(self.model)
                # use accelerator.save to be safe in distributed
                self.accelerator.save(unwrapped.state_dict(), self.checkpoint_path)
                self.accelerator.print(f"Validation loss improved: saved to {self.checkpoint_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1


            # --- Early stopping ---
            if epochs_no_improve >= self.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            # --- ReduceLROnPlateau (manual) ---
            if self.scheduler_name == "step" and len(history["val_loss"]) > self.lr_reduction_patience:
                if lr_plateau_counter > self.lr_reduction_patience and history["val_loss"][-1] >= history["val_loss"][-(self.lr_reduction_patience + 1)]:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = max(old_lr * self.lr_reduction_factor, self.min_lr)
                    if new_lr < old_lr:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        print(f"ReduceLROnPlateau: lr reduced from {old_lr:.1e} to {new_lr:.1e}")
                        lr_plateau_counter = 0
                else:
                    lr_plateau_counter += 1

            if self.scheduler:
                if self.scheduler_step_mode == "epoch":
                    self.scheduler.step()
                elif self.scheduler_step_mode == "plateau":
                    self.scheduler.step(history["val_loss"][-1])

        self.accelerator.end_training()
        return history
