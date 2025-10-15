import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from typing import Tuple
import random
from scipy import ndimage

class DataAugmentation:
    """Data augmentation utilities for 3D volumes."""
    
    @staticmethod
    def rotate_volume(volume: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rotate volume by random angle.
        
        Parameters
        ----------
            volume: 3D volume tensor
            
        Returns
        ----------
            Rotated volume
        """
        def _scipy_rotate(vol):
            angles = [-20, -10, -5, 5, 10, 20]
            angle = random.choice(angles)
            rotated = rotated = ndimage.rotate(vol, angle, axes=(1, 2), reshape=False, order=1)
            rotated = np.clip(rotated, 0, 1)
            return rotated.astype(np.float32)
        rotated = _scipy_rotate(volume)
        return torch.from_numpy(rotated)
    
    @staticmethod
    def augment_training(volume: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation for training data.
        
        Parameters
        ----------
            volume: 3D volume
            label: Classification label
            
        Returns
        ----------
            Augmented volume and label
        """
        volume = DataAugmentation.rotate_volume(volume)
        volume = volume.unsqueeze(0)
        return volume
    
    @staticmethod
    def prepare_validation(volume: torch.Tensor) -> torch.Tensor:
        """
        Prepare validation data (no augmentation).
        
        Parameters
        ----------
            volume: 3D volume
            label: Classification label
            
        Returns
        ----------
            Volume with channel dimension and label
        """
        volume = volume.unsqueeze(0)
        return volume