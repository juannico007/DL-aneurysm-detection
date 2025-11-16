from .model import AneurysmDetectionModel
from .data_augmentation import DataAugmentation, rotate_batch_gpu
from sklearn.metrics import accuracy_score, f1_score, fbeta_score
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from typing import List, Sequence, Tuple
import torch.nn.functional as F
import bitsandbytes as bnb
from pathlib import Path
from accelerate import Accelerator

import h5py


def custom_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def _round_up(n: int, m: int = 16) -> int:
    return ((n + m - 1) // m) * m

def pad_collate_3d(batch: List[Tuple[torch.Tensor, torch.Tensor]], multiple: int = 16):
    """
    Each x in batch has shape (C=1, D, H, W) and sizes may differ.
    We pad on the right to (Dt,Ht,Wt) = per-batch max (rounded to 'multiple'),
    then stack into (B, 1, Dt, Ht, Wt).
    """
    xs, ys = zip(*batch)  # xs: tuple of tensors (1,D,H,W), ys: tuple of ints/tensors

    # ensure labels tensor (B,)
    Y = torch.as_tensor(ys, dtype=torch.long)

    # get per-batch target size
    shapes = [x.shape[-3:] for x in xs]  # (D,H,W)
    Dm = max(s[0] for s in shapes)
    Hm = max(s[1] for s in shapes)
    Wm = max(s[2] for s in shapes)
    Dt, Ht, Wt = _round_up(Dm, multiple), _round_up(Hm, multiple), _round_up(Wm, multiple)

    padded = []
    for x in xs:
        # x is (1,D,H,W)
        _, D, H, W = x.shape
        pd, ph, pw = Dt - D, Ht - H, Wt - W
        x = x.to(torch.float16)
        # pad order: (W_left, W_right, H_left, H_right, D_left, D_right)
        x_pad = F.pad(x, (0, pw, 0, ph, 0, pd), value=0.0)
        padded.append(x_pad)

    X = torch.stack(padded, dim=0)  # (B,1,Dt,Ht,Wt)
    return X, Y

class AneurysmDataset(Dataset):
    """Lightweight dataset that fetches volumes from HDF5 on demand."""

    def __init__(
        self,
        h5_path: Path,
        series_ids: Sequence[str],
        labels: Sequence[int],
        transform=None,
    ):
        self.h5_path = str(h5_path)
        self.series_ids = list(series_ids)
        self.labels = [int(label) for label in labels]
        self.transform = transform
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

    def __getitem__(self, idx):
        handle = self._get_file()
        pid = self.series_ids[idx]
        group = handle["series"][pid]
        volume = torch.from_numpy(group["vol"][:])

        if self.transform:
            volume = self.transform(volume)
        elif volume.ndim == 3:
            volume = volume.unsqueeze(0)

        label = self.labels[idx]
        return volume, label

class TrainingPipeline:
    """Training pipeline for the aneurysm detection model."""
    
    def __init__(
        self,
        model: AneurysmDetectionModel,
        batch_size: int = 4,
        epochs: int = 4,
        learning_rate : float = 1e-4,
        early_stopping_patience: int = 20,
        lr_reduction_patience: int = 10,
        lr_reduction_factor : float = 0.5,
        min_lr : float = 1e-7,
        checkpoint_path: str = "aneurysm_detection_best.pt",
        mixed_precision: str = "fp16",
        grad_accum_steps: int = 2,
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
        self.early_stopping_patience = early_stopping_patience
        self.lr_reduction_patience = lr_reduction_patience
        self.lr_reduction_factor = lr_reduction_factor
        self.min_lr = min_lr
        self.checkpoint_path = checkpoint_path

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision, gradient_accumulation_steps=grad_accum_steps
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
    
    def _safe_gather(self, tensor):
        """Safely gather tensors across processes for Gloo backend."""
        tensor = tensor.flatten().contiguous()
        
        # Get max size across processes
        local_size = torch.tensor(tensor.shape[0], device=self.device)
        all_sizes = self.accelerator.gather(local_size)
        max_size = all_sizes.max().item()
        
        # Pad to max size
        current_size = tensor.shape[0]
        if current_size == max_size:
            return self.accelerator.gather(tensor)
        
        tensor = torch.nn.functional.pad(tensor, (0, max_size - current_size), value=0)

        return self.accelerator.gather(tensor)
    
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
            weight_decay=0.01)
        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.96) #Exponential decay scheduler

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        #Since we have to define our own training loop, we have to keep track
        #of these variables for best model, early stopping and learning rate decrease
        best_val_acc = 0
        epochs_no_improve = 0
        lr_plateau_counter = 0

        #Also we want to keep track of our training history
        history = {
            "train_loss": [], "train_acc": [], "train_f1": [], "train_f2": [],
            "val_loss": [], "val_acc": [], "val_f1": [], "val_f2": []
        }

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_losses, all_preds, all_labels = [], [], []

            for x_batch, y_batch in tqdm(self.train_loader):
                # y to float column vector
                y_batch = y_batch.float().unsqueeze(1)

                x_batch = x_batch.to(self.device, dtype=torch.float16, non_blocking=True)
                y_batch = y_batch.to(self.device, dtype=torch.float16, non_blocking=True)

                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        x_batch = rotate_batch_gpu(x_batch)
                        outputs = self.model(x_batch)
                        loss = self.criterion(outputs, y_batch)

                    self.optimizer.zero_grad(set_to_none=True)
                    self.accelerator.backward(loss)
                    self.optimizer.step()

                train_losses.append(self.accelerator.gather(loss.detach()).mean().item())
                # gather predictions/labels only for metrics
                all_preds.append(outputs.detach())
                all_labels.append(y_batch.detach())

            all_preds = torch.cat(all_preds, dim=0).to(self.device, dtype=torch.float32).contiguous()
            all_labels = torch.cat(all_labels, dim=0).to(self.device, dtype=torch.float32).contiguous()

            # Now safe to gather on GPU
            all_preds  = self._safe_gather(all_preds)
            all_labels = self._safe_gather(all_labels)

            # Post-process on CPU
            probs = torch.sigmoid(all_preds)
            train_preds_np = (probs >= 0.5).int().cpu().numpy().ravel()
            train_labels_np  = all_labels.int().cpu().numpy().ravel()

            train_acc = accuracy_score(train_labels_np, train_preds_np)
            train_f1 = f1_score(train_labels_np, train_preds_np, zero_division=0)
            train_f2 = fbeta_score(train_labels_np, train_preds_np, beta=2, zero_division=0)

            history["train_loss"].append(np.mean(train_losses))
            history["train_acc"].append(train_acc)
            history["train_f1"].append(train_f1)
            history["train_f2"].append(train_f2)

            # --- Validation ---
            self.model.eval()
            val_losses = []
            val_preds_list, val_labels_list = [], []
            with torch.no_grad():
                for x_batch, y_batch in self.val_loader:
                    y_batch = y_batch.float().unsqueeze(1)
                    with self.accelerator.autocast():
                        outputs = self.model(x_batch)
                        loss = self.criterion(outputs, y_batch)
                    val_losses.append(self.accelerator.gather(loss.detach()).mean().item())

                    # Gather across processes
                    preds = self._safe_gather(outputs.detach().to(torch.float32))
                    labels = self._safe_gather(y_batch.detach().to(torch.float32))

                    # Sigmoid + threshold (done in float32 for numerical stability)
                    probs = torch.sigmoid(preds)
                    preds_np = (probs >= 0.5).int().cpu().numpy()
                    labs_np  = labels.int().cpu().numpy()

                    val_preds_list.append(preds_np)
                    val_labels_list.append(labs_np)

                    # Free temporary tensors
                    del preds, labels, probs, preds_np, labs_np, outputs, loss
                    torch.cuda.empty_cache()

            val_preds_np = np.concatenate(val_preds_list, axis=0).ravel()
            val_labels_np = np.concatenate(val_labels_list, axis=0).ravel()

            val_acc = accuracy_score(val_labels_np, val_preds_np)
            val_f1 = f1_score(val_labels_np, val_preds_np, zero_division=0)
            val_f2 = fbeta_score(val_labels_np, val_preds_np, beta=2, zero_division=0)
            history["val_loss"].append(np.mean(val_losses))
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
            history["val_f2"].append(val_f2)

            self.accelerator.print(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {history['train_loss'][-1]:.4f} | Acc: {train_acc:.4f} | "
                f"F1: {train_f1:.4f} | F2: {train_f2:.4f} | "
                f"Val Loss: {history['val_loss'][-1]:.4f} | Acc: {val_acc:.4f} | "
                f"F1: {val_f1:.4f} | F2: {val_f2:.4f}"
            )

            epoch_metrics = {
                "train/loss": history["train_loss"][-1],
                "train/acc": train_acc,
                "train/f1": train_f1,
                "train/f2": train_f2,
                "val/loss": history["val_loss"][-1],
                "val/acc": val_acc,
                "val/f1": val_f1,
                "val/f2": val_f2,
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
            if len(history["val_loss"]) > self.lr_reduction_patience:
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

            self.scheduler.step()

        self.accelerator.end_training()
        return history
