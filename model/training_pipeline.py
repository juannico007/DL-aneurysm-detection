from .model import AneurysmDetectionModel
from .data_augmentation import DataAugmentation
from sklearn.metrics import accuracy_score
from typing import Tuple
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from torchvision import transforms
import numpy as np

class CustomDataset(Dataset):
    """Custom dataset to perform dataloading in pytorch"""
    def __init__(self, x_data, y_data, transform=None):
        """
        Initialize dataset
        
        Parameters
        ----------
            x_data: Patients volumes
            y_data: labels
            transform: Desired transformation for the data
        """
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        # Convert numpy to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Apply transforms / augmentation
        if self.transform:
            x = self.transform(x)

        return x, y

class TrainingPipeline:
    """Training pipeline for the aneurysm detection model."""
    
    def __init__(
        self,
        model: AneurysmDetectionModel,
        batch_size: int = 2,
        epochs: int = 100,
        learning_rate : float = 1e-4,
        early_stopping_patience: int = 20,
        lr_reduction_patience: int = 10,
        lr_reduction_factor : float = 0.5,
        min_lr : float = 1e-7,
        checkpoint_path: str = "aneurysm_detection_best.pt"
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
    
    def create_dataloaders(
        self,
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
        train_dataset = CustomDataset(x_train, y_train, transform=train_transforms)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,          # same as tf shuffle
            num_workers=4,         # similar to tf AUTOTUNE parallel calls
            pin_memory=True        # speeds up transfer to GPU
        )
        
        val_transforms = transforms.Compose([
            DataAugmentation.prepare_validation
        ])
        val_dataset = CustomDataset(x_val, y_val, transform=val_transforms)

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,          # same as tf shuffle
            num_workers=4,         # similar to tf AUTOTUNE parallel calls
            pin_memory=True        # speeds up transfer to GPU
        )
        
        return train_loader, val_loader
    
    def train(
        self,
        train_loader: TensorDataset,
        val_loader: TensorDataset
    ) -> dict:
        """
        Train the model.
        
        Parameters
        ----------
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns
        ----------
            Training history
        """
        # Check if GPU available
        if torch.cuda.is_available():
            print("GPU is on")
            self.device = "cuda"
            self.model = self.model.to("cuda")
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.96) #Exponential decay scheduler

        #Since we have to define our own training loop, we have to keep track
        #of these variables for best model, early stopping and learning rate decrease
        best_val_acc = 0
        epochs_no_improve = 0
        lr_plateau_counter = 0

        #Also we want to keep track of our training history
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []
        }

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_losses, all_preds, all_labels = [], [], []

            for x_batch, y_batch in tqdm(train_loader):
                if torch.cuda.is_available:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device).float().unsqueeze(1)
                else:
                    y_batch = y_batch.float().unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()  # Step scheduler per batch

                train_losses.append(loss.item())
                if torch.cuda.is_available:
                    outputs = outputs.detach().cpu()
                    y_batch = y_batch.detach().cpu()
                all_preds.extend(outputs.numpy())
                all_labels.extend(y_batch.numpy())

            # --- Training metrics ---
            train_preds = (np.array(all_preds) > 0.5).astype(int)
            train_acc = accuracy_score(all_labels, train_preds)

            history["train_loss"].append(np.mean(train_losses))
            history["train_acc"].append(train_acc)

            # --- Validation ---
            self.model.eval()
            val_losses, val_preds, val_labels = [], [], []

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    if torch.cuda.is_available():
                        x_batch = x_batch.to(self.device)
                        y_batch = y_batch.to(self.device).float().unsqueeze(1)
                    else:
                        y_batch.float().unsqueeze(1)
                    outputs = self.model(x_batch)
                    loss = self.criterion(outputs, y_batch)
                    val_losses.append(loss.item())
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(y_batch.cpu().numpy())

            val_preds_bin = (np.array(val_preds) > 0.5).astype(int)
            val_acc = accuracy_score(val_labels, val_preds_bin)

            history["val_loss"].append(np.mean(val_losses))
            history["val_acc"].append(val_acc)

            print(f"Epoch {epoch}/{self.epochs} | "
                  f"Train Loss: {np.mean(train_losses):.4f} | Acc: {train_acc:.4f} | "
                  f"Val Loss: {np.mean(val_losses):.4f} | Acc: {val_acc:.4f}")

            # --- Checkpointing ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"Validation accuracy improved: model saved to {self.checkpoint_path}")
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

        return history