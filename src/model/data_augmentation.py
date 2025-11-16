import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from typing import Tuple
import random
from scipy import ndimage
import math, random
import torch.nn.functional as F
# class DataAugmentation:
#     """Data augmentation utilities for 3D volumes."""
    
#     @staticmethod
#     def rotate_volume(volume: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Rotate volume by random angle.
        
#         Parameters
#         ----------
#             volume: 3D volume tensor
            
#         Returns
#         ----------
#             Rotated volume
#         """
#         def _scipy_rotate(vol):
#             angles = [-20, -10, -5, 5, 10, 20]
#             angle = random.choice(angles)
#             rotated = rotated = ndimage.rotate(vol, angle, axes=(1, 2), reshape=False, order=1)
#             rotated = np.clip(rotated, 0, 1)
#             return rotated.astype(np.float32)
#         rotated = _scipy_rotate(volume)
#         return torch.from_numpy(rotated)
    
#     @staticmethod
#     def augment_training(volume: torch.Tensor) -> torch.Tensor:
#         """
#         Apply augmentation for training data.
        
#         Parameters
#         ----------
#             volume: 3D volume
#             label: Classification label
            
#         Returns
#         ----------
#             Augmented volume and label
#         """
#         volume = DataAugmentation.rotate_volume(volume)
#         volume = volume.unsqueeze(0)
#         return volume
    
#     @staticmethod
#     def prepare_validation(volume: torch.Tensor) -> torch.Tensor:
#         """
#         Prepare validation data (no augmentation).
        
#         Parameters
#         ----------
#             volume: 3D volume
#             label: Classification label
            
#         Returns
#         ----------
#             Volume with channel dimension and label
#         """
#         volume = volume.unsqueeze(0)
#         return volume
class DataAugmentation:
    """Lightweight CPU pre-processing for 3D data."""

    @staticmethod
    def augment_training(volume: torch.Tensor) -> torch.Tensor:
        # No rotation here — done in batch GPU stage
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)  # (1, D, H, W)
        return volume.to(torch.float16)   # keep CPU fp16

    @staticmethod
    def prepare_validation(volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        return volume.to(torch.float16)


def rotate_batch_gpu(x: torch.Tensor, angles: torch.Tensor = None, mode: str = 'bilinear'):
    B, C, D, H, W = x.shape
    if angles is None:
        angles = torch.empty(B, device=x.device).uniform_(-20, 20) * (math.pi / 180)
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    theta = torch.zeros(B, 3, 4, device=x.device, dtype=x.dtype)
    theta[:,0,0] = cos_a;  theta[:,0,1] = -sin_a
    theta[:,1,0] = sin_a;  theta[:,1,1] =  cos_a
    theta[:,2,2] = 1.0
    grid = F.affine_grid(theta, size=x.size(), align_corners=False)
    x_rot = F.grid_sample(x, grid, mode=mode, padding_mode='zeros', align_corners=False)
    return x_rot, angles