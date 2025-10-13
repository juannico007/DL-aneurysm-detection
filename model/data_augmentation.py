from scipy import ndimage
import tensorflow as tf
import random
import numpy as np
from typing import Tuple

class DataAugmentation:
    """Data augmentation utilities for 3D volumes."""
    
    @staticmethod
    def rotate_volume(volume: tf.Tensor) -> tf.Tensor:
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
            rotated = ndimage.rotate(vol, angle, reshape=False)
            rotated = np.clip(rotated, 0, 1)
            return rotated.astype(np.float32)
        
        return tf.numpy_function(_scipy_rotate, [volume], tf.float32)
    
    @staticmethod
    def augment_training(volume: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
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
        volume = tf.expand_dims(volume, axis=-1)
        return volume, label
    
    @staticmethod
    def prepare_validation(volume: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
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
        volume = tf.expand_dims(volume, axis=-1)
        return volume, label