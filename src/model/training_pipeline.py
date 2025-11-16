from email import utils
from .unet import UNet
from .data_augmentation import DataAugmentation, rotate_batch_gpu
from sklearn.metrics import accuracy_score, f1_score, fbeta_score
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

import h5py


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
    ):
        self.h5_path = str(h5_path)
        self.series_ids = list(series_ids)
        self.labels = [int(label) for label in labels]
        self.transform = transform
        self.radius = radius
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
        mask_np = np.zeros(shape, dtype=np.uint8)
        centers = self.localizer_points.get(str(pid))
        if centers is not None:
            for center in centers:
                mask_np |= make_sphere_mask(shape, center, radius=self.radius)
        return torch.from_numpy(mask_np).unsqueeze(0)

    @staticmethod
    def _empty_mask(shape: Tuple[int, int, int]) -> torch.Tensor:
        return torch.zeros((1,) + tuple(shape), dtype=torch.uint8)

    def __getitem__(self, idx):
        handle = self._get_file()
        pid = self.series_ids[idx]
        group = handle["series"][pid]
        volume = torch.from_numpy(group["vol"][:])
        mask = (
            self._build_mask(pid, volume.shape[-3:])
            if self.localizer_points.get(str(pid))
            else self._empty_mask(volume.shape[-3:])
        )

        if self.transform:
            volume = self.transform(volume)
        elif volume.ndim == 3:
            volume = volume.unsqueeze(0)

        label = self.labels[idx]
        return volume, mask, label

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
        localizer_csv: Optional[Path] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
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
        self.localizer_csv = Path(localizer_csv) if localizer_csv else None
        self.scheduler_config = dict(scheduler_config) if scheduler_config else {
            "name": "step",
            "step_size": 100,
            "gamma": 0.96,
        }

        self.scheduler_step_mode: Optional[str] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.scheduler_name: Optional[str] = None

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision, 
            gradient_accumulation_steps=grad_accum_steps
        )
    
    def create_dataloaders(
        self,
        h5_path: Path,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create pytorch DataLoaders.
        
        Parameters
        ----------
            x_train: Training volumes
            y_train: Training labels
            x_val: Validation volumes
            y_val: Validation labels
            
        Returns
        ----------
            Tuple of (train_dataset, val_dataset)
        """
        train_transforms = transforms.Compose([
            DataAugmentation.augment_training
        ])
        train_dataset = AneurysmDataset(
            h5_path=h5_path,
            series_ids=x_train,
            labels=y_train,
            transform=train_transforms,
            localizer_csv=self.localizer_csv,
            radius=self.radius,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,          # same as tf shuffle
            num_workers=0,         # similar to tf AUTOTUNE parallel calls
            pin_memory=True,        # speeds up transfer to GPU
            collate_fn=pad_collate_3d,
        )
        
        val_transforms = transforms.Compose([
            DataAugmentation.prepare_validation
        ])
        val_dataset = AneurysmDataset(
            h5_path=h5_path,
            series_ids=x_val,
            labels=y_val,
            transform=val_transforms,
            localizer_csv=self.localizer_csv,
            radius=self.radius,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,          # same as tf shuffle
            num_workers=0,         # similar to tf AUTOTUNE parallel calls
            pin_memory=True,        # speeds up transfer to GPU
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
        max_size = all_sizes.max().item()
        
        # Pad to max size
        current_size = tensor.shape[0]
        if current_size < max_size:
            tensor = torch.nn.functional.pad(tensor, (0, max_size - current_size), value=0)

        gathered = self.accelerator.gather(tensor)

        if max_size == 0:
            return gathered

        masks = [
            (torch.arange(max_size, device=self.device) < size)
            for size in all_sizes.tolist()
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
        self.criterion = nn.DiceLoss()

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self._configure_scheduler()
        #Since we have to define our own training loop, we have to keep track
        #of these variables for best model, early stopping and learning rate decrease
        best_val_acc = 0
        epochs_no_improve = 0
        lr_plateau_counter = 0

        #Also we want to keep track of our training history
        history = {
            "train_loss": [], "train_acc": [], "train_f1": [], "train_f2": [],
            "val_loss": [], "val_acc": [], "val_f1": [], "val_f2": [],
            "train_dice": [], "val_dice": []
        }

        eps = 1e-6
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_losses = []
            train_scan_preds, train_scan_labels, train_dice_scores = [], [], []

            for x_batch, mask_batch, _ in tqdm(self.train_loader):
                x_batch = x_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                mask_batch = mask_batch.to(self.device, dtype=torch.float16, non_blocking=True)

                x_batch, angles = rotate_batch_gpu(x_batch, mode='bilinear')
                mask_batch, _ = rotate_batch_gpu(mask_batch, angles=angles, mode='nearest')
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        x_batch = rotate_batch_gpu(x_batch)
                        outputs = self.model(x_batch)
                        loss = self.criterion(outputs, mask_batch)

                    self.optimizer.zero_grad(set_to_none=True)
                    self.accelerator.backward(loss)
                    utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    if self.scheduler and self.scheduler_step_mode == "batch":
                        self.scheduler.step()

                train_losses.append(self.accelerator.gather(loss.detach()).mean().item())

                with torch.no_grad():
                    probs = torch.sigmoid(outputs.detach().to(torch.float32))
                    targets = mask_batch.detach().to(torch.float32)
                    pred_masks = (probs >= 0.5).float()
                    scan_pred = pred_masks.flatten(start_dim=1).any(dim=1).float()
                    scan_label = targets.flatten(start_dim=1).any(dim=1).float()
                    train_scan_preds.append(scan_pred)
                    train_scan_labels.append(scan_label)

                    pred_sum = pred_masks.sum(dim=(1, 2, 3, 4))
                    target_sum = targets.sum(dim=(1, 2, 3, 4))
                    intersection = (pred_masks * targets).sum(dim=(1, 2, 3, 4))
                    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
                    train_dice_scores.append(dice)

            train_scan_preds = torch.cat(train_scan_preds, dim=0).to(self.device, dtype=torch.float32)
            train_scan_labels = torch.cat(train_scan_labels, dim=0).to(self.device, dtype=torch.float32)
            train_dice_scores = torch.cat(train_dice_scores, dim=0).to(self.device, dtype=torch.float32)

            train_scan_preds = self._safe_gather(train_scan_preds)
            train_scan_labels = self._safe_gather(train_scan_labels)
            train_dice_scores = self._safe_gather(train_dice_scores)

            train_preds_np = train_scan_preds.cpu().numpy().astype(int)
            train_labels_np = train_scan_labels.cpu().numpy().astype(int)
            train_dice_mean = train_dice_scores.cpu().numpy().mean() if train_dice_scores.numel() > 0 else 0.0

            train_acc = accuracy_score(train_labels_np, train_preds_np)
            train_f1 = f1_score(train_labels_np, train_preds_np, zero_division=0)
            train_f2 = fbeta_score(train_labels_np, train_preds_np, beta=2, zero_division=0)

            history["train_loss"].append(np.mean(train_losses))
            history["train_acc"].append(train_acc)
            history["train_f1"].append(train_f1)
            history["train_f2"].append(train_f2)
            history["train_dice"].append(train_dice_mean)

            # --- Validation ---
            self.model.eval()
            val_losses = []
            val_scan_preds, val_scan_labels, val_dice_scores = [], [], []
            with torch.no_grad():
                for x_batch, mask_batch, _ in self.val_loader:
                    x_batch = x_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                    mask_batch = mask_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                    with self.accelerator.autocast():
                        outputs = self.model(x_batch)
                        loss = self.criterion(outputs, mask_batch)
                    val_losses.append(self.accelerator.gather(loss.detach()).mean().item())

                    probs = torch.sigmoid(outputs.detach().to(torch.float32))
                    targets = mask_batch.detach().to(torch.float32)
                    pred_masks = (probs >= 0.5).float()
                    scan_pred = pred_masks.flatten(start_dim=1).any(dim=1).float()
                    scan_label = targets.flatten(start_dim=1).any(dim=1).float()
                    val_scan_preds.append(scan_pred)
                    val_scan_labels.append(scan_label)

                    pred_sum = pred_masks.sum(dim=(1, 2, 3, 4))
                    target_sum = targets.sum(dim=(1, 2, 3, 4))
                    intersection = (pred_masks * targets).sum(dim=(1, 2, 3, 4))
                    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
                    val_dice_scores.append(dice)

                    del outputs, loss
                    torch.cuda.empty_cache()

            val_scan_preds = torch.cat(val_scan_preds, dim=0).to(self.device, dtype=torch.float32)
            val_scan_labels = torch.cat(val_scan_labels, dim=0).to(self.device, dtype=torch.float32)
            val_dice_scores = torch.cat(val_dice_scores, dim=0).to(self.device, dtype=torch.float32)

            val_scan_preds = self._safe_gather(val_scan_preds)
            val_scan_labels = self._safe_gather(val_scan_labels)
            val_dice_scores = self._safe_gather(val_dice_scores)

            val_preds_np = val_scan_preds.cpu().numpy().astype(int)
            val_labels_np = val_scan_labels.cpu().numpy().astype(int)
            val_dice_mean = val_dice_scores.cpu().numpy().mean() if val_dice_scores.numel() > 0 else 0.0

            val_acc = accuracy_score(val_labels_np, val_preds_np)
            val_f1 = f1_score(val_labels_np, val_preds_np, zero_division=0)
            val_f2 = fbeta_score(val_labels_np, val_preds_np, beta=2, zero_division=0)
            history["val_loss"].append(np.mean(val_losses))
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
            history["val_f2"].append(val_f2)
            history["val_dice"].append(val_dice_mean)

            self.accelerator.print(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {history['train_loss'][-1]:.4f} | Acc: {train_acc:.4f} | "
                f"F1: {train_f1:.4f} | F2: {train_f2:.4f} | Dice: {history['train_dice'][-1]:.4f} | "
                f"Val Loss: {history['val_loss'][-1]:.4f} | Acc: {val_acc:.4f} | "
                f"F1: {val_f1:.4f} | F2: {val_f2:.4f} | Dice: {history['val_dice'][-1]:.4f}"
            )

            epoch_metrics = {
                "train/loss": history["train_loss"][-1],
                "train/acc": train_acc,
                "train/f1": train_f1,
                "train/f2": train_f2,
                "train/dice": history["train_dice"][-1],
                "val/loss": history["val_loss"][-1],
                "val/acc": val_acc,
                "val/f1": val_f1,
                "val/f2": val_f2,
                "val/dice": history["val_dice"][-1],
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
            self.accelerator.log(epoch_metrics, step=epoch)

            # --- Checkpointing ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # unwrap for vanilla state_dict saving
                unwrapped = self.accelerator.unwrap_model(self.model)
                # use accelerator.save to be safe in distributed
                self.accelerator.save(unwrapped.state_dict(), self.checkpoint_path)
                self.accelerator.print(f"Validation accuracy improved: saved to {self.checkpoint_path}")
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
