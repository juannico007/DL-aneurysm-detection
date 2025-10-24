from pathlib import Path
import pandas as pd
import numpy as np
import itk
from typing import Tuple

class ScanDataLoader:
    """Custom data loader for volumes."""
    
    def __init__(self, data_dir: Path, csv_path: Path, cache_dir: Path = Path("cache")):
        """
        Initialize the data loader.
        
        Parameters
        ----------
            data_dir: Directory containing .nii.gz files
            csv_path: Path to CSV file with SeriesInstanceUID and Aneurysm Present columns
            cache_dir: Directory for caching processed volumes
        """
        self.data_dir = Path(data_dir)
        self.csv_path = Path(csv_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.image_paths = []
        self.labels = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset paths and labels from CSV."""
        df = pd.read_csv(self.csv_path)
        
        for uid, label in zip(df["SeriesInstanceUID"], df["Aneurysm Present"].values):
            path = self.data_dir / f"{uid}.npz"
            if path.exists():
                self.image_paths.append(str(path))
                self.labels.append(int(label))
        
        print(f"Loaded {len(self.image_paths)} scans: "
              f"{sum(self.labels)} positive, {len(self.labels) - sum(self.labels)} negative")
    
    def _load_volume(self, path: str) -> np.ndarray:
        """
        Load a single volume from disk with caching.
        
        Parameters
        ----------
            path: Path to .nii.gz file
            
        Returns
        ----------
            3D numpy array of the volume
        """
        cache_path = self.cache_dir / (Path(path).stem + ".npy")
        
        if cache_path.exists():
            return np.load(cache_path)
        
        image = itk.imread(path, itk.F)
        volume = itk.GetArrayFromImage(image).astype(np.float32)
        
        np.save(cache_path, volume)
        print(f"Cached: {Path(path).name}")
        
        return volume
    
    # def load_all_volumes(self) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Load all volumes into memory.
        
    #     Returns
    #     ----------
    #         Tuple of (volumes, labels) as numpy arrays
    #     """
    #     volumes = []
    #     for path in self.image_paths:
    #         volume = self._load_volume(path)
    #         volumes.append(volume)
        
    #     return np.array(volumes, dtype=object), np.array(self.labels)
    
    # def split_data(self, train_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Split data into training and validation sets.
        
    #     Parameters
    #     ----------
    #         train_ratio: Ratio of data to use for training
            
    #     Returns
    #     ----------
    #         Tuple of (x_train, y_train, x_val, y_val)
    #     """
    #     volumes, labels = self.load_all_volumes()
        
    #     # Separate positive and negative samples
    #     positive_idx = np.where(labels == 1)[0]
    #     negative_idx = np.where(labels == 0)[0]
        
    #     split_pos = int(train_ratio * len(positive_idx))
    #     split_neg = int(train_ratio * len(negative_idx))
        
    #     train_idx = np.concatenate([positive_idx[:split_pos], negative_idx[:split_neg]])
    #     val_idx = np.concatenate([positive_idx[split_pos:], negative_idx[split_neg:]])
        
    #     np.random.shuffle(train_idx)
    #     np.random.shuffle(val_idx)
        
    #     x_train, y_train = volumes[train_idx], labels[train_idx]
    #     x_val, y_val = volumes[val_idx], labels[val_idx]
        
    #     print(f"\nDataset split:")
    #     print(f"  Training: {len(x_train)} samples "
    #           f"(pos: {np.sum(y_train)}, neg: {len(y_train) - np.sum(y_train)})")
    #     print(f"  Validation: {len(x_val)} samples "
    #           f"(pos: {np.sum(y_val)}, neg: {len(y_val) - np.sum(y_val)})")
        
    #     return x_train, y_train, x_val, y_val

    def split_data(self, train_ratio: float = 0.7):
        df = pd.read_csv(self.csv_path)
        paths, labels = [], []
        for uid, label in zip(df["SeriesInstanceUID"], df["Aneurysm Present"]):
            path = self.data_dir / f"{uid}.npz"
            if path.exists():
                paths.append(str(path))
                labels.append(int(label))

        labels = np.array(labels)
        positive_idx = np.where(labels == 1)[0]
        negative_idx = np.where(labels == 0)[0]

        split_pos = int(train_ratio * len(positive_idx))
        split_neg = int(train_ratio * len(negative_idx))

        train_idx = np.concatenate([positive_idx[:split_pos], negative_idx[:split_neg]])
        val_idx = np.concatenate([positive_idx[split_pos:], negative_idx[split_neg:]])

        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)

        x_train = [paths[i] for i in train_idx]
        y_train = labels[train_idx]
        x_val = [paths[i] for i in val_idx]
        y_val = labels[val_idx]

        print(f"\nDataset split:")
        print(f"  Training: {len(x_train)} samples "
              f"(pos: {np.sum(y_train)}, neg: {len(y_train) - np.sum(y_train)})")
        print(f"  Validation: {len(x_val)} samples "
              f"(pos: {np.sum(y_val)}, neg: {len(y_val) - np.sum(y_val)})")
        
        return x_train, y_train, x_val, y_val

