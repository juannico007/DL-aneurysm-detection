import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from typing import Tuple
import random
from scipy import ndimage
import math, random
import torch.nn.functional as F

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


def augment_batch_gpu(
    x: torch.Tensor, 
    coords: torch.Tensor = None,
    angles: torch.Tensor = None, 
    flips: torch.Tensor = None,
    mode: str = 'bilinear',
    flip_prob: float = 0.5,
    intensity_prob: float = 0.5,
    noise_prob: float = 0.5
):
    """
    Apply GPU-accelerated augmentation to a batch of volumes and optional coordinates.
    
    Args:
        x: (B, C, D, H, W) input volume
        coords: (B, 3) optional coordinates in [0, 1] range (z, y, x)
        angles: (B,) optional rotation angles in radians
        flips: (B, 3) optional flip masks (0 or 1)
        mode: interpolation mode for grid_sample
        flip_prob: probability of flipping along each axis
        intensity_prob: probability of applying intensity scaling
        noise_prob: probability of adding Gaussian noise
        
    Returns:
        x_aug: Augmented volume
        coords_aug: Augmented coordinates (if coords provided, else None)
        angles: The angles used
        flips: The flips used
    """
    B, C, D, H, W = x.shape
    device = x.device
    dtype = x.dtype
    
    # --- 1. Rotation ---
    if angles is None:
        # Random rotation between -20 and 20 degrees
        angles = torch.empty(B, device=device).uniform_(-20, 20) * (math.pi / 180)
        
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    
    theta = torch.zeros(B, 3, 4, device=device, dtype=dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 2, 2] = 1.0
    
    # Apply rotation to volume
    grid = F.affine_grid(theta, size=x.size(), align_corners=False)
    x_aug = F.grid_sample(x, grid, mode=mode, padding_mode='zeros', align_corners=False)
    
    # Apply rotation to coordinates
    coords_aug = None
    if coords is not None:
        # coords are (z, y, x) in [0, 1] range.
        # We need to convert to (x, y, z) in [-1, 1] range for matrix multiplication.
        # Note: PyTorch grid uses (x, y, z) order.
        
        # 1. Expand to (B, 3, 1)
        # Swap z,y,x to x,y,z
        c_z = coords[:, 0]
        c_y = coords[:, 1]
        c_x = coords[:, 2]
        
        # Normalize to [-1, 1]
        # (val * 2) - 1
        p_x = (c_x * 2.0) - 1.0
        p_y = (c_y * 2.0) - 1.0
        p_z = (c_z * 2.0) - 1.0
        
        # Stack as (B, 3, 1) vector: [x, y, z]
        p_vec = torch.stack([p_x, p_y, p_z], dim=1).unsqueeze(2) # (B, 3, 1)
        
        # 2. Apply Inverse Rotation
        # The theta matrix maps Output -> Input.
        # P_in = Theta @ P_out
        # We want to find P_out given P_in.
        # P_out = Theta^-1 @ P_in
        # Since Theta is a rotation (orthogonal), Theta^-1 = Theta^T
        
        # Extract 3x3 rotation part
        rot_mat = theta[:, :3, :3] # (B, 3, 3)
        
        # Inverse (Transpose)
        rot_inv = rot_mat.transpose(1, 2) # (B, 3, 3)
        
        # Cast to float32 for coordinate transformation (coords are always float32)
        rot_inv = rot_inv.to(torch.float32)
        
        # Apply inverse rotation
        p_rot = torch.bmm(rot_inv, p_vec) # (B, 3, 1)
        
        # 3. Convert back to [0, 1] and (z, y, x)
        p_rot = p_rot.squeeze(2) # (B, 3)
        
        # Un-normalize: (val + 1) / 2
        new_x = (p_rot[:, 0] + 1.0) / 2.0
        new_y = (p_rot[:, 1] + 1.0) / 2.0
        new_z = (p_rot[:, 2] + 1.0) / 2.0
        
        coords_aug = torch.stack([new_z, new_y, new_x], dim=1)
        
    # --- 2. Random Flipping ---
    # Generate random flip masks (B, 3)
    # 0: no flip, 1: flip
    if flips is None:
        if flip_prob > 0:
            flips = (torch.rand(B, 3, device=device) < flip_prob).int()
        else:
            flips = torch.zeros(B, 3, device=device, dtype=torch.int)
        
    for b in range(B):
        dims_to_flip = []
        # When indexing x_aug[b], shape is (C, D, H, W), so dimensions are 0, 1, 2, 3
        if flips[b, 0]: dims_to_flip.append(1) # Depth (D is dim 1 in x_aug[b])
        if flips[b, 1]: dims_to_flip.append(2) # Height (H is dim 2 in x_aug[b])
        if flips[b, 2]: dims_to_flip.append(3) # Width (W is dim 3 in x_aug[b])
        
        if dims_to_flip:
            x_aug[b] = torch.flip(x_aug[b], dims=dims_to_flip)
            
            if coords_aug is not None:
                # coords are (z, y, x)
                # If flipped D (z), z -> 1-z
                if flips[b, 0]: coords_aug[b, 0] = 1.0 - coords_aug[b, 0]
                if flips[b, 1]: coords_aug[b, 1] = 1.0 - coords_aug[b, 1]
                if flips[b, 2]: coords_aug[b, 2] = 1.0 - coords_aug[b, 2]

    # --- 3. Intensity Scaling ---
    if intensity_prob > 0:
        # Random scale factor [0.8, 1.2]
        scales = torch.rand(B, 1, 1, 1, 1, device=device) * 0.4 + 0.8
        # Apply only to selected batch elements
        apply_mask = (torch.rand(B, 1, 1, 1, 1, device=device) < intensity_prob).float()
        
        # x_new = x * scale * mask + x * (1-mask)
        # Simplified: x * (1 + (scale-1)*mask)
        factor = 1.0 + (scales - 1.0) * apply_mask
        x_aug = x_aug * factor

    # --- 4. Gaussian Noise ---
    if noise_prob > 0:
        # Noise sigma [0, 0.1]
        noise_sigma = 0.05
        apply_mask = (torch.rand(B, 1, 1, 1, 1, device=device) < noise_prob).float()
        
        noise = torch.randn_like(x_aug) * noise_sigma
        x_aug = x_aug + noise * apply_mask

    # Ensure output dtype matches input dtype (grid_sample promotes fp16 to fp32)
    x_aug = x_aug.to(dtype)
    
    return x_aug, coords_aug, angles, flips