from .model import AneurysmDetectionModel
from .data_augmentation import DataAugmentation
from typing import Tuple
import tensorflow as tf
import numpy as np
import keras

class TrainingPipeline:
    """Training pipeline for the aneurysm detection model."""
    
    def __init__(
        self,
        model: AneurysmDetectionModel,
        batch_size: int = 2,
        epochs: int = 100,
        checkpoint_path: str = "aneurysm_detection_best.keras"
    ):
        """
        Initialize training pipeline.
        
        Parameters
        ----------
            model: AneurysmDetectionModel instance
            batch_size: Batch size for training
            epochs: Number of training epochs
            checkpoint_path: Path to save best model
        """
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
    
    def create_datasets(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create TensorFlow datasets.
        
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
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = (
            train_dataset
            .shuffle(buffer_size=len(x_train))
            .map(DataAugmentation.augment_training, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = (
            val_dataset
            .map(DataAugmentation.prepare_validation, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        return train_dataset, val_dataset
    
    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset
    ) -> keras.callbacks.History:
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
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"Batch size: {self.batch_size}")
        print(f"Model checkpoint: {self.checkpoint_path}\n")
        
        history = self.model.get_model().fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history