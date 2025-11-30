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

class MultiTaskLoss(nn.Module):
    def __init__(self, tasks: List[str]):
        super().__init__()
        self.tasks = tasks
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1)) for task in tasks
        })

    def forward(self, loss_dict: Dict[str, torch.Tensor]):
        total_loss = 0
        for task, loss in loss_dict.items():
            if task not in self.log_vars:
                continue
            log_var = self.log_vars[task]
            precision = torch.exp(-log_var)
            total_loss += 0.5 * precision * loss + 0.5 * log_var
        return total_loss

class TrainingWrapper(nn.Module):
    """
    Wrapper to bundle the main model and the multi-task loss module 
    so they can be prepared together by Accelerate/DeepSpeed.
    """
    def __init__(self, model, multi_task_loss):
        super().__init__()
        self.model = model
        self.multi_task_loss = multi_task_loss
    
    def forward(self, x):
        return self.model(x)

class UnifiedCurriculumDice(nn.Module):
    def __init__(self, beta=2.0, lambda_tail=1.0, lambda_max=0.0, epsilon=1e-6):
        super().__init__()
        self.beta = beta
        self.lambda_tail = lambda_tail
        self.lambda_max = lambda_max
        self.epsilon = epsilon

    def forward(self, pred, target, alpha):
        # 1. Define the spatial switch (Mask)
        # alpha is the curriculum threshold
        core_mask = (target >= alpha).float()
        tail_mask = 1.0 - core_mask

        # 2. Calculate Core Components
        # We only look at Intersection inside the Gaussian Core
        p_core = pred * core_mask
        t_core = target * core_mask
        
        # Intersection (Numerator)
        intersection = torch.sum(p_core * t_core, dim=[1, 2, 3, 4])
        
        # Core Disagreement (L1 Error inside the core)
        diff_core = torch.sum(torch.abs(p_core - t_core), dim=[1, 2, 3, 4])

        # 3. Calculate Tail Components
        # We replace standard error with your polynomial suppression
        # Logic: If pred < alpha, error is 0. If pred > alpha, error scales polynomially.
        tail_penalty = torch.relu(pred - alpha).pow(self.beta)
        weighted_tail_error = torch.sum(tail_penalty * tail_mask, dim=[1, 2, 3, 4])

        # 4. The Unified Semimetric Formula
        # L = 1 - (2*Int) / (2*Int + Core_Diff + Lambda * Tail_Error)
        
        denominator = (2 * intersection) + diff_core + (self.lambda_tail * weighted_tail_error) + self.epsilon
        
        dice_score = (2 * intersection + self.epsilon) / denominator
        
        loss = 1.0 - dice_score.mean()

        # 5. Background Max Suppression (L_infinity on background)
        if self.lambda_max > 0:
            # We want to penalize the single highest prediction in the background
            # pred * tail_mask gives us predictions in the background (and 0 in core)
            # We take the max over spatial dimensions for each batch item
            tail_probs = pred * tail_mask
            max_tail_error = tail_probs.view(tail_probs.size(0), -1).max(dim=1)[0]
            loss += self.lambda_max * max_tail_error.mean()
            
        return loss

def custom_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def custom_collate_with_coords(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    coords = [item[3] for item in batch]
    
    target = torch.stack(target)
    labels = torch.LongTensor(labels)
    coords = torch.stack(coords)
    
    return [data, target, labels, coords]

def _round_up(n: int, m: int = 16) -> int:
    return ((n + m - 1) // m) * m

def pad_collate_3d(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], multiple: int = 16):
    """
    Each x in batch has shape (C=1, D, H, W) and sizes may differ.
    We pad on the right to (Dt,Ht,Wt) = per-batch max (rounded to 'multiple'),
    then stack into (B, 1, Dt, Ht, Wt).
    """
    xs, ms, ys, cs = zip(*batch)  # xs: tuple of tensors (1,D,H,W), ys: tuple of ints/tensors

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
    C = torch.stack(cs, dim=0)
    return X, M, Y, C

def make_sphere_mask(shape, center, radius=5.0):
    """Return a binary 3D mask with a filled sphere."""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cy, cx = center
    dist = np.square(x - cx)+ np.square(y - cy) + np.square(z - cz)
    mask = dist <= radius**2
    return mask.astype(np.uint8)

def _gaussian_heatmap(
        shape: Tuple[int, int, int], 
        center: Tuple[float, float, float], 
        sigma: float) -> np.ndarray:
    """Create a single 3D Gaussian heatmap with peak 1.0, optimized with bounding box."""
    heatmap = np.zeros(shape, dtype=np.float32)
    cz, cy, cx = center
    
    # Define bounding box (3 sigma rule covers >99% of mass)
    radius = int(math.ceil(3 * sigma))
    
    z_min = max(0, int(math.floor(cz - radius)))
    z_max = min(shape[0], int(math.ceil(cz + radius)) + 1)
    
    y_min = max(0, int(math.floor(cy - radius)))
    y_max = min(shape[1], int(math.ceil(cy + radius)) + 1)
    
    x_min = max(0, int(math.floor(cx - radius)))
    x_max = min(shape[2], int(math.ceil(cx + radius)) + 1)
    
    if z_min >= z_max or y_min >= y_max or x_min >= x_max:
        return heatmap

    # Create grid only for the bounding box
    z, y, x = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
    
    dist2 = np.square(x - cx) + np.square(y - cy) + np.square(z - cz)
    heatmap[z_min:z_max, y_min:y_max, x_min:x_max] = np.exp(-dist2 / (2.0 * sigma ** 2))
    
    return heatmap


def make_heatmaps(
        shape: Tuple[int, int, int], 
        center: Tuple[float, float, float], 
        sigmas: Sequence[float]) -> np.ndarray:
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

# class AneurysmDataset(Dataset):
#     """Lightweight dataset that fetches volumes (and optional masks) from HDF5 on demand."""

#     def __init__(
#         self,
#         h5_path: Path,
#         series_ids: Sequence[str],
#         labels: Sequence[int],
#         transform=None,
#         localizer_csv: Optional[Path] = None,
#         radius: float = 5.0,
#         heatmap_sizes: Sequence[float] = (15.0,),
#         suppress_tau: float = 0.3,
#     ):
#         self.h5_path = str(h5_path)
#         self.series_ids = list(series_ids)
#         self.labels = [int(label) for label in labels]
#         self.transform = transform
#         self.radius = radius
#         self.heatmap_sizes = list(heatmap_sizes) if heatmap_sizes else [radius]
#         self.localizer_points = self._load_localizer_points(localizer_csv)
#         self._h5 = None
#         self.suppress_tau = suppress_tau

#     def __len__(self):
#         return len(self.series_ids)

#     def _get_file(self):
#         if self._h5 is None:
#             self._h5 = h5py.File(self.h5_path, "r")
#         return self._h5

#     def close(self):
#         if self._h5 is not None:
#             self._h5.close()
#             self._h5 = None

#     def __del__(self):
#         self.close()

#     # returns a dict mapping SeriesInstanceUID to np.ndarray of shape (N, 3) with (z_new,y_new,x_new) points
#     def _load_localizer_points(self, csv_path: Optional[Path]) -> Dict[str, np.ndarray]:
#         if not csv_path:
#             return {}
#         csv_path = Path(csv_path).expanduser()
#         if not csv_path.exists():
#             print(f"[Warning] Localizer CSV not found: {csv_path}. Masks disabled.")
#             return {}
#         df = pd.read_csv(csv_path)
#         required = {"SeriesInstanceUID", "x_new", "y_new", "z_new"}
#         if not required.issubset(df.columns):
#             print(
#                 "[Warning] Localizer CSV missing required columns "
#                 f"({', '.join(sorted(required))}). Masks disabled."
#             )
#             return {}
#         grouped = {}
#         for uid, group in df.groupby("SeriesInstanceUID"):
#             grouped[str(uid)] = group[["z_new", "y_new", "x_new"]].to_numpy(dtype=float)
#         return grouped

#     def _build_mask(self, pid: str, shape: Tuple[int, int, int]) -> torch.Tensor:
#         mask_np = np.zeros((len(self.heatmap_sizes),) + shape, dtype=np.float32)
#         centers = self.localizer_points.get(str(pid))
#         if centers is not None:
#             for center in centers:
#                 heatmaps = make_heatmaps(shape, center, sigmas=self.heatmap_sizes, tau_gauss=self.suppress_tau)
#                 mask_np = np.maximum(mask_np, heatmaps)
#         return torch.from_numpy(mask_np)

#     @staticmethod
#     def _empty_mask(shape: Tuple[int, int, int], channels: int = 1) -> torch.Tensor:
#         return torch.zeros((channels,) + tuple(shape), dtype=torch.float32)

#     def __getitem__(self, idx):
#         handle = self._get_file()
#         pid = self.series_ids[idx]
#         group = handle["series"][pid]
#         volume = torch.from_numpy(group["vol"][:])
#         centers = self.localizer_points.get(str(pid))
#         if centers is not None and len(centers):
#             mask = self._build_mask(pid, volume.shape[-3:])
#         else:
#             mask = self._empty_mask(volume.shape[-3:], channels=len(self.heatmap_sizes))

#         if self.transform:
#             volume = self.transform(volume)
#         elif volume.ndim == 3:
#             volume = volume.unsqueeze(0)

#         label = self.labels[idx]
#         return volume, mask, label


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
            try:
                label = int(row["label"])
            except (TypeError, ValueError):
                # Skip rows that look like headers or malformed entries
                print(f"[Warning] Skipping malformed row in patches CSV: {row}")
                continue
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
        
        # Calculate relative coordinates
        rel_coords = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32) # Default for background
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
            
            # Normalize center to [0, 1] for regression
            # center is (z, y, x) relative to patch
            rel_coords = torch.tensor([
                center[0] / self.patch_size,
                center[1] / self.patch_size,
                center[2] / self.patch_size
            ], dtype=torch.float32)
            
            mask_np = make_heatmaps(patch_np.shape, center, sigmas=self.heatmap_sizes)

        volume = torch.from_numpy(patch_np)
        mask = torch.from_numpy(mask_np)

        if self.transform:
            volume = self.transform(volume)
        elif volume.ndim == 3:
            volume = volume.unsqueeze(0)

        return volume, mask, record.label, rel_coords

class TrainingPipeline:
    """Training pipeline for the aneurysm detection model."""
    
    def __init__(
        self,
        model: UNet,
        batch_size: int = 4,
        epochs: int = 4,
        learning_rate : float = 1e-4,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 40,
        lr_reduction_patience: int = 10,
        lr_reduction_factor : float = 0.5,
        min_lr : float = 1e-7,
        checkpoint_path: str = "aneurysm_detection_best.pt",
        mixed_precision: str = "fp16",
        grad_accum_steps: int = 2,
        radius: float = 5.0,
        heatmap_sigma: float = 15.0,
        heatmap_decay: Optional[Dict[str, Any]] = None,
        heatmap_min_sigma: float = 5.0,
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
        self.heatmap_min_sigma = float(heatmap_min_sigma)
        self.heatmap_decay_config = dict(heatmap_decay) if heatmap_decay else {"name": "epoch", "epoch": 0}
        if "min_sigma" in self.heatmap_decay_config:
            self.heatmap_min_sigma = float(self.heatmap_decay_config["min_sigma"])
        self.heatmap_decay_epoch = int(self.heatmap_decay_config.get("epoch", 0))
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
        self.alpha_max = float(lw.get("alpha_max", 0.5))
        self.beta = float(lw.get("beta", 2.0))
        self.lambda_tail = float(lw.get("lambda_tail", 1.0))
        self.lambda_max = float(lw.get("lambda_max", 0.0))
        self.seg_weight = float(lw.get("segmentation", 1.0))
        self.cls_weight = float(lw.get("classification", 0.5))
        self.coord_weight = float(lw.get("coordinates", 1.0))
        self.background_weight = float(lw.get("background", 0.1))
        
        self.neg_warmup_epochs = int(neg_warmup_epochs)
        # Default ramp epochs if not provided (though we expect it from CLI now)
        # If neg_ramp_epochs is passed in loss_weights or kwargs, use it, else default to remaining epochs
        self.neg_ramp_epochs = int(lw.get("neg_ramp_epochs", max(1, self.epochs - self.neg_warmup_epochs)))
        self.alpha_min = float(lw.get("alpha_min", 0.05))
        self.alpha_schedule = lw.get("alpha_schedule", "cosine")

        self.scheduler_step_mode: Optional[str] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.scheduler_name: Optional[str] = None

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision, 
            gradient_accumulation_steps=grad_accum_steps
        )
        
        # Initialize loss function
        self.criterion = UnifiedCurriculumDice(
            beta=self.beta, 
            lambda_tail=self.lambda_tail,
            lambda_max=self.lambda_max
        )

        # Identify active tasks for Multi-Task Learning
        self.tasks = ["segmentation"]
        if self.cls_weight > 0:
            self.tasks.append("classification")
        if self.coord_weight > 0:
            self.tasks.append("coordinates")
        
        self.multi_task_loss = MultiTaskLoss(self.tasks)

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
        cfg = self.heatmap_decay_config
        name = cfg.get("name", "epoch")
        if name == "epoch":
            new_sigma = self._sigma_for_epoch(epoch)
            if new_sigma != self.current_heatmap_sigma:
                self.accelerator.print(f"[Heatmap] Sigma update at epoch {epoch}: {self.current_heatmap_sigma} -> {new_sigma}")
                self._set_heatmap_sigma(new_sigma)
        elif name == "plateau":
            # handled in training loop after validation using metrics
            return
        elif name == "threshold":
            # handled in training loop after validation using metrics
            return
        else:
            return

    def _neg_penalty_scale(self, epoch: int) -> float:
        """Gaussian ramp from 0 to 1 after warmup for negative suppression terms."""
        if self.neg_ramp_epochs <= 0:
            return 1.0
        t = (epoch - self.neg_warmup_epochs) / max(1, self.neg_ramp_epochs)
        t = min(max(t, 0.0), 1.0)
        return math.exp(-5.0 * (1.0 - t) * (1.0 - t))

    def _get_current_alpha(self, epoch: int) -> float:
        """Calculate alpha for the current epoch based on the curriculum schedule."""
        # 1. Warmup phase
        if epoch <= self.neg_warmup_epochs:
            return self.alpha_max
            
        # 2. Ramp phase
        # Epochs since warmup ended (1-based index relative to ramp start)
        ramp_step = epoch - self.neg_warmup_epochs
        
        if ramp_step > self.neg_ramp_epochs:
            # 3. Stabilization phase
            return self.alpha_min
            
        # Calculate progress t from 0 to 1
        t = (ramp_step - 1) / max(1, self.neg_ramp_epochs - 1)
        t = min(max(t, 0.0), 1.0)
        
        if self.alpha_schedule == "cosine":
            # Cosine decay from 1 to 0
            decay = 0.5 * (1.0 + math.cos(t * math.pi))
        else:
            # Linear decay (fallback)
            decay = 1.0 - t
            
        # Map decay (1->0) to alpha (alpha_max -> alpha_min)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * decay

    def _maybe_decay_heatmap_dynamic(self, metrics: Dict[str, float], epoch: int):
        cfg = self.heatmap_decay_config
        name = cfg.get("name", "epoch")
        if name == "plateau":
            metric_name = cfg.get("metric", "val_loss")
            mode = cfg.get("mode", "min")
            patience = int(cfg.get("patience", 3))
            factor = float(cfg.get("factor", 0.5))
            if not hasattr(self, "_hd_best"):
                self._hd_best = None
                self._hd_wait = 0
            current = metrics.get(metric_name)
            if current is None:
                return
            if self._hd_best is None:
                self._hd_best = current
                self._hd_wait = 0
                return
            improved = (current < self._hd_best) if mode == "min" else (current > self._hd_best)
            if improved:
                self._hd_best = current
                self._hd_wait = 0
            else:
                self._hd_wait += 1
                if self._hd_wait >= patience:
                    new_sigma = max(self.current_heatmap_sigma * factor, self.heatmap_min_sigma)
                    if new_sigma < self.current_heatmap_sigma:
                        self.accelerator.print(f"[Heatmap] Plateau decay at epoch {epoch}: {self.current_heatmap_sigma} -> {new_sigma}")
                        self.current_heatmap_sigma = new_sigma
                    self._hd_wait = 0
        elif name == "threshold":
            metric_name = cfg.get("metric", "val_peak_err_series")
            mode = cfg.get("mode", "min")
            thresh = float(cfg.get("threshold", 0.0))
            factor = float(cfg.get("factor", 0.5))
            current = metrics.get(metric_name)
            if current is None:
                return
            hit = (current <= thresh) if mode == "min" else (current >= thresh)
            if hit:
                new_sigma = max(self.current_heatmap_sigma * factor, self.heatmap_min_sigma)
                if new_sigma < self.current_heatmap_sigma:
                    self.accelerator.print(f"[Heatmap] Threshold decay at epoch {epoch}: {self.current_heatmap_sigma} -> {new_sigma}")
                    self.current_heatmap_sigma = new_sigma

    def compute_background_penalty(self, pred, target, input_volume):
        """
        Compute penalty for high predictions on background voxels.
        
        Args:
            pred: (B, 1, D, H, W) - predicted probabilities
            target: (B, 1, D, H, W) - ground truth mask
            input_volume: (B, 1, D, H, W) - z-scored input volume
            
        Returns:
            background_penalty: scalar tensor
        """
        # Identify background: z-scored values < 0
        background_mask = (input_volume < 0).float()
        
        # Only consider true background (not in ground truth)
        true_background = background_mask * (1 - target)
        
        # Penalize high predictions on background
        if true_background.sum() > 0:
            background_penalty = (pred * true_background).sum() / (true_background.sum() + 1e-6)
        else:
            background_penalty = torch.tensor(0.0, device=pred.device)
            
        return background_penalty

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

        # Wrap model and multi_task_loss together for DeepSpeed compatibility
        self.wrapper = TrainingWrapper(self.model, self.multi_task_loss)
        
        self.optimizer = bnb.optim.Adam8bit(
            self.wrapper.parameters(), 
            lr=self.learning_rate,
            betas = (0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay)

        
        self.criterion = UnifiedCurriculumDice(
            beta=self.beta,
            lambda_tail=self.lambda_tail
        )
        
        cls_pos_weight = getattr(self, "pos_weight", torch.tensor(1.0))
        cls_pos_weight = cls_pos_weight.to(self.device)
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight)
        
        self.wrapper, self.optimizer = self.accelerator.prepare(
            self.wrapper, self.optimizer
        )
        # Update references to point to wrapped components
        self.model = self.wrapper.model
        self.multi_task_loss = self.wrapper.multi_task_loss
        self._configure_scheduler()
        #Since we have to define our own training loop, we have to keep track
        #of these variables for best model, early stopping and learning rate decrease
        best_val_loss = float("inf")
        epochs_no_improve = 0
        lr_plateau_counter = 0

        #Also we want to keep track of our training history
        history = {
            "train_loss": [], 
            "train_acc": [], 
            "train_sensitivity": [], 
            "train_ppv": [], 
            "train_npv": [],
            "train_fpr": [],
            "train_specificity": [],
            "train_fnr": [],
            "train_mse": [],
            "val_loss": [], 
            "val_acc": [], 
            "val_sensitivity": [], 
            "val_ppv": [], 
            "val_npv": [],
            "train_dice": [], 
            "val_dice": [],
            "val_fpr": [],
            "val_specificity": [],
            "val_fnr": [],
            "val_mse": [],
        }

        eps = 1e-6
        for epoch in range(1, self.epochs + 1):
            self._maybe_decay_heatmap(epoch)
            current_alpha = self._get_current_alpha(epoch)
            # Get current MTL weights for display
            mtl_info = []
            for task, param in self.multi_task_loss.log_vars.items():
                weight = torch.exp(-param).item()
                mtl_info.append(f"{task}={weight:.4f}")
            mtl_str = ", ".join(mtl_info)
            self.accelerator.print(f"Epoch {epoch}: Alpha = {current_alpha:.4f}, Sigma = {self.current_heatmap_sigma:.1f}, Weights: [{mtl_str}]")
            
            self.model.train()
            train_losses = []
            train_dice_scores = []
            train_mse_scores = [] 
            train_core_list, train_peak_list, train_focus_list = [], [], []
            all_preds, all_labels = [], []
            train_neg_mean_list, train_neg_max_list = [], []
            train_seg_losses, train_cls_losses, train_coord_losses = [], [], []
            train_total_losses = []
            for x_batch, mask_batch, y_batch, coord_batch in tqdm(self.train_loader):
                x_batch = x_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                mask_batch = mask_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                y_batch = y_batch.float().unsqueeze(1)
                y_batch = y_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                coord_batch = coord_batch.to(self.device, dtype=torch.float32, non_blocking=True) # Coords are float32
                
                # Note: Rotation augmentation needs to handle coordinates too if we want to be correct.
                # For now, let's assume no rotation or that we accept the small error if rotation is small.
                # Ideally, we should rotate the coordinates.
                # TODO: Implement coordinate rotation.
                
                x_batch, angles = rotate_batch_gpu(x_batch, mode='bilinear')
                mask_batch, _ = rotate_batch_gpu(mask_batch, angles=angles, mode='nearest')
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        outputs = self.model(x_batch)
                        # outputs[0] is segmentation, outputs[1] is classification (if any)
                        # We only use segmentation output for the new loss
                        seg_logits = outputs[0]
                        seg_probs = torch.sigmoid(seg_logits)
                        
                        # Target for segmentation is the heatmap (mask_batch)
                        # Assuming single channel output for now or matching channels
                        targets = mask_batch[:, self.primary_heatmap_index:self.primary_heatmap_index+1]
                        
                        seg_loss = self.criterion(seg_probs, targets, current_alpha)
                        
                        # Classification loss
                        cls_loss = torch.tensor(0.0, device=self.device)
                        if self.cls_weight > 0 and len(outputs) > 1:
                            cls_logits = outputs[1]
                            cls_loss = self.classification_loss(cls_logits, y_batch)
                            
                        # Background penalty loss
                        if self.background_weight > 0:
                            bg_penalty = self.compute_background_penalty(seg_probs, targets, x_batch)
                            # Treat background penalty as part of segmentation task
                            seg_loss += self.background_weight * bg_penalty
                        
                        # Combine losses using Multi-Task Uncertainty Weights
                        loss_dict = {"segmentation": seg_loss}
                        
                        if self.cls_weight > 0 and len(outputs) > 1:
                            loss_dict["classification"] = cls_loss
                            
                        if self.coord_weight > 0 and len(outputs) > 2:
                            reg_output = outputs[2].float() # Ensure float32
                            # coord_batch is (B, 3)
                            # Calculate MSE loss per sample
                            mse_loss = F.mse_loss(reg_output, coord_batch, reduction='none').mean(dim=1)
                            
                            # Scale by patch_size to get voxel-scale errors
                            scaled_mse = self.patch_size * mse_loss
                            
                            # Masking: Only penalize if label == 1
                            # y_batch is (B, 1)
                            mask = y_batch.squeeze(1)
                            masked_mse = (scaled_mse * mask).sum() / (mask.sum() + 1e-6)
                            
                            # Use the scaled masked_mse for the coordinate task
                            loss_dict["coordinates"] = masked_mse
                            
                        loss = self.multi_task_loss(loss_dict)

                        self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        max_norm = 5
                        unclipped_global_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                        # self.accelerator.print(
                        #         f"[GRAD] Max Norm: {max_norm:.2f} | Unclipped Global Norm: {unclipped_global_norm:.4f}"
                        #     )
                    
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.scheduler and self.scheduler_step_mode == "batch":
                            self.scheduler.step()

                # Track individual task losses
                train_seg_losses.append(self.accelerator.gather(seg_loss.detach()).mean().item())
                if "classification" in loss_dict:
                    train_cls_losses.append(self.accelerator.gather(cls_loss.detach()).mean().item())
                if "coordinates" in loss_dict:
                    train_coord_losses.append(self.accelerator.gather(masked_mse.detach()).mean().item())
                train_total_losses.append(self.accelerator.gather(loss.detach()).mean().item())

                #Dice calculation (soft dice on soft targets)
                with torch.no_grad():
                    mask_probs = torch.sigmoid(outputs[0].detach().to(torch.float32))
                    targets = mask_batch[:, self.primary_heatmap_index:self.primary_heatmap_index+1].detach().to(torch.float32)
                    mse_value = F.mse_loss(mask_probs, targets).item()
                    train_mse_scores.append(mse_value)
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
                        tgt_bin = (targets_pos >= 0.05).float()
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

            train_mse_mean = np.mean(train_mse_scores)
            
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
            train_sens = tp / max(tp + fn, 1)          # sensitivity/recall
            train_ppv = tp / max(tp + fp, 1)           # precision/PPV
            train_npv = tn / max(tn + fn, 1)           # NPV
            train_fpr = fp / max(tn + fp, 1)           # false positive rate
            train_specificity = tn / max(tn + fp, 1)   # true negative rate
            train_fnr = fn / max(fn + tp, 1)           # false positive rate

            history["train_loss"].append(np.mean(train_losses))
            history["train_acc"].append(train_acc)
            history["train_sensitivity"].append(train_sens)
            history["train_ppv"].append(train_ppv)
            history["train_npv"].append(train_npv)
            history["train_fpr"].append(train_fpr)
            history["train_specificity"].append(train_specificity)
            history["train_fnr"].append(train_fnr)
            history["train_dice"].append(train_dice_mean)
            
            # Track individual task losses (use safe mean to avoid numpy warnings)
            train_seg_loss_mean = sum(train_seg_losses) / len(train_seg_losses) if train_seg_losses else 0.0
            train_cls_loss_mean = sum(train_cls_losses) / len(train_cls_losses) if train_cls_losses else 0.0
            train_coord_loss_mean = sum(train_coord_losses) / len(train_coord_losses) if train_coord_losses else 0.0
            train_total_loss_mean = sum(train_total_losses) / len(train_total_losses) if train_total_losses else 0.0
            history.setdefault("train_seg_loss", []).append(train_seg_loss_mean)
            history.setdefault("train_cls_loss", []).append(train_cls_loss_mean)
            history.setdefault("train_coord_loss", []).append(train_coord_loss_mean)
            history.setdefault("train_total_loss", []).append(train_total_loss_mean)
            history.setdefault("train_core_dice_series", []).append(train_core_dice_mean)
            history.setdefault("train_peak_err_series", []).append(train_peak_mean)
            history.setdefault("train_focus_series", []).append(train_focus_mean)
            history.setdefault("train_neg_prob_mean_series", []).append(train_neg_prob_mean)
            history.setdefault("train_neg_prob_max_series", []).append(train_neg_prob_max)

            # --- Validation ---
            self.model.eval()
            val_losses = []
            val_dice_scores = []
            val_mse_list = []
            val_preds_list, val_labels_list = [], []
            with torch.no_grad():
                val_losses = []
                val_core_list, val_peak_list, val_focus_list = [], [], []
                val_neg_mean_list, val_neg_max_list = [], []
                val_dice_scores = []
                val_seg_losses, val_cls_losses, val_coord_losses = [], [], []
                val_total_losses = []
                for x_batch, mask_batch, y_batch, coord_batch in self.val_loader:
                    x_batch = x_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                    mask_batch = mask_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                    y_batch = y_batch.float().unsqueeze(1)
                    y_batch = y_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                    coord_batch = coord_batch.to(self.device, dtype=torch.float32, non_blocking=True)
                    with self.accelerator.autocast():
                        outputs = self.model(x_batch)
                        seg_logits = outputs[0]
                        seg_probs = torch.sigmoid(seg_logits)
                        targets = mask_batch[:, self.primary_heatmap_index:self.primary_heatmap_index+1]
                        
                        # Use current alpha for validation loss too? Or a fixed one?
                        # Usually validation loss should be consistent, but if it's a curriculum loss, 
                        # maybe we should use the same alpha as training to see convergence.
                        # Or maybe alpha=0 (hardest) to see true performance?
                        # Let's use current_alpha for consistency with training objective.
                        seg_loss = self.criterion(seg_probs, targets, current_alpha)
                        
                        cls_loss = torch.tensor(0.0, device=self.device)
                        if self.cls_weight > 0 and len(outputs) > 1:
                            cls_logits = outputs[1]
                            cls_loss = self.classification_loss(cls_logits, y_batch)
                            
                        # Background penalty (for validation consistency)
                        if self.background_weight > 0:
                            bg_penalty = self.compute_background_penalty(seg_probs, targets, x_batch)
                            seg_loss += self.background_weight * bg_penalty

                        loss_dict = {"segmentation": seg_loss}
                        if self.cls_weight > 0 and len(outputs) > 1:
                            loss_dict["classification"] = cls_loss
                        
                        if self.coord_weight > 0 and len(outputs) > 2:
                            reg_output = outputs[2].float()
                            mse_loss = F.mse_loss(reg_output, coord_batch, reduction='none').mean(dim=1)
                            
                            # Scale by patch_size to get voxel-scale errors
                            scaled_mse = self.patch_size * mse_loss
                            
                            mask = y_batch.squeeze(1)
                            masked_mse = (scaled_mse * mask).sum() / (mask.sum() + 1e-6)
                            loss_dict["coordinates"] = masked_mse
                            
                        loss = self.multi_task_loss(loss_dict)
                        
                    # Track individual task losses
                    val_seg_losses.append(self.accelerator.gather(seg_loss.detach()).mean().item())
                    if "classification" in loss_dict:
                        val_cls_losses.append(self.accelerator.gather(cls_loss.detach()).mean().item())
                    if "coordinates" in loss_dict:
                        val_coord_losses.append(self.accelerator.gather(masked_mse.detach()).mean().item())
                    val_total_losses.append(self.accelerator.gather(loss.detach()).mean().item())

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
                    
                    val_mse_value = F.mse_loss(mask_probs, targets)
                    val_mse_list.append(val_mse_value)
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
            val_mse_tensor = torch.stack(val_mse_list).to(self.device, dtype=torch.float32)
            val_mse_tensor = self._safe_gather(val_mse_tensor)
            val_mse_mean = val_mse_tensor.cpu().numpy().mean() if val_mse_tensor.numel() > 0 else 0.0
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
            val_fpr = fp / max(tn + fp, 1)         
            val_specificity = tn / max(tn + fp, 1)   
            val_fnr = fn / max(fn + tp, 1)
            val_loss = np.mean(val_losses)
            
            # Calculate mean task losses (use safe mean to avoid numpy warnings)
            val_seg_loss_mean = sum(val_seg_losses) / len(val_seg_losses) if val_seg_losses else 0.0
            val_cls_loss_mean = sum(val_cls_losses) / len(val_cls_losses) if val_cls_losses else 0.0
            val_coord_loss_mean = sum(val_coord_losses) / len(val_coord_losses) if val_coord_losses else 0.0
            val_total_loss_mean = sum(val_total_losses) / len(val_total_losses) if val_total_losses else 0.0
            # Use segmentation loss as primary validation metric
            val_loss = val_seg_loss_mean
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_sensitivity"].append(val_sens)
            history["val_ppv"].append(val_ppv)
            history["val_npv"].append(val_npv)
            history["val_fpr"].append(val_fpr)
            history["val_specificity"].append(val_specificity)
            history["val_fnr"].append(val_fnr)
            history["val_dice"].append(val_dice_mean)
            history.setdefault("val_seg_loss", []).append(val_seg_loss_mean)
            history.setdefault("val_cls_loss", []).append(val_cls_loss_mean)
            history.setdefault("val_coord_loss", []).append(val_coord_loss_mean)
            history.setdefault("val_total_loss", []).append(val_total_loss_mean)
            history.setdefault("val_core_dice_series", []).append(val_core_dice_mean)
            history.setdefault("val_peak_err_series", []).append(val_peak_mean)
            history.setdefault("val_focus_series", []).append(val_focus_mean)
            history.setdefault("val_neg_prob_mean_series", []).append(val_neg_prob_mean)
            history.setdefault("val_neg_prob_max_series", []).append(val_neg_prob_max)
            history.setdefault("alpha_series", []).append(current_alpha)
            history.setdefault("heatmap_sigma_series", []).append(self.current_heatmap_sigma)
            
            # Log Multi-Task Weights (sigma^2 = exp(s))
            mtl_weights = {}
            for task, param in self.multi_task_loss.log_vars.items():
                sigma2 = torch.exp(param).item()
                weight = 1.0 / sigma2
                mtl_weights[f"mtl_weight/{task}"] = weight
                history.setdefault(f"mtl_weight_{task}", []).append(weight)
            if self.accelerator.is_main_process:
                train_seg_loss_mean = history["train_seg_loss"][-1]
                train_cls_loss_mean = history["train_cls_loss"][-1]
                train_coord_loss_mean = history["train_coord_loss"][-1]
                train_total_loss_mean = history["train_total_loss"][-1]
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]
                
                # Create table header with dynamic variables
                self.accelerator.print(f"\n{'='*150}")
                self.accelerator.print(f"Epoch {epoch}/{self.epochs} | LR: {current_lr:.2e} | Alpha: {current_alpha:.4f} | Sigma: {self.current_heatmap_sigma:.1f} | Weights: [{mtl_str}]")
                self.accelerator.print(f"{'='*150}")
                
                # Metrics table
                self.accelerator.print(f"{'':8} | {'TotalLoss':>9} | {'SegLoss':>8} | {'ClsLoss':>8} | {'CoordLoss':>9} | {'Acc':>6} | {'Sens':>6} | {'FPR':>6} | {'Spec':>6} | {'FNR':>6} | {'PPV':>6} | {'NPV':>6} | {'Dice':>6} | {'Core':>6} | {'Peak':>6} | {'Focus':>6} | {'NegMean':>7} | {'NegMax':>7}")
                self.accelerator.print(f"{'-'*150}")
                self.accelerator.print(
                    f"{'Train':8} | {train_total_loss_mean:9.4f} | {train_seg_loss_mean:8.4f} | {train_cls_loss_mean:8.4f} | {train_coord_loss_mean:9.4f} | "
                    f"{train_acc:6.4f} | {train_sens:6.4f} | {train_fpr:6.4f} | {train_specificity:6.4f} | {train_fnr:6.4f} | {train_ppv:6.4f} | {train_npv:6.4f} | "
                    f"{train_dice_mean:6.4f} | {train_core_dice_mean:6.4f} | {train_peak_mean:6.2f} | "
                    f"{train_focus_mean:6.4f} | {train_neg_prob_mean:7.4f} | {train_neg_prob_max:7.4f}"
                )
                self.accelerator.print(
                    f"{'Val':8} | {val_total_loss_mean:9.4f} | {val_seg_loss_mean:8.4f} | {val_cls_loss_mean:8.4f} | {val_coord_loss_mean:9.4f} | "
                    f"{val_acc:6.4f} | {val_sens:6.4f} | {val_fpr:6.4f} | {val_specificity:6.4f} | {val_fnr:6.4f} | {val_ppv:6.4f} | {val_npv:6.4f} | "
                    f"{val_dice_mean:6.4f} | {val_core_dice_mean:6.4f} | {val_peak_mean:6.2f} | "
                    f"{val_focus_mean:6.4f} | {val_neg_prob_mean:7.4f} | {val_neg_prob_max:7.4f}"
                )
                self.accelerator.print(f"{'='*150}\n")
            

            epoch_metrics = {
                "train/seg_loss": train_seg_loss_mean,
                "train/cls_loss": train_cls_loss_mean,
                "train/coord_loss": train_coord_loss_mean,
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
                "val/seg_loss": val_seg_loss_mean,
                "val/cls_loss": val_cls_loss_mean,
                "val/coord_loss": val_coord_loss_mean,
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
            # Add MTL weights to logs
            for k, v in mtl_weights.items():
                epoch_metrics[k] = v
            self.accelerator.log(epoch_metrics, step=epoch)

            unwrapped_wrapper = self.accelerator.unwrap_model(self.wrapper)
            unwrapped = unwrapped_wrapper.model
            # per-epoch checkpoint (optional)
            ckpt_path = Path(self.checkpoint_path)
            epoch_ckpt = ckpt_path.with_name(f"{epoch}_{ckpt_path.name}")
            epoch_ckpt.parent.mkdir(parents=True, exist_ok=True)
            self.accelerator.save(unwrapped.state_dict(), epoch_ckpt)
            # --- Checkpointing ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # unwrap for vanilla state_dict saving
                # use accelerator.save to be safe in distributed
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                self.accelerator.save(unwrapped.state_dict(), self.checkpoint_path)
                self.accelerator.print(f"Validation loss improved: saved to {self.checkpoint_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Dynamic heatmap decay based on metrics/config
            self._maybe_decay_heatmap_dynamic(
                {
                    "val_loss": val_loss,
                    "val_core_dice": val_core_dice_mean,
                    "val_peak_err": val_peak_mean,
                },
                epoch,
            )


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
