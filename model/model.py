from typing import Tuple
import tensorflow as tf
import keras
from keras import layers

class AneurysmDetectionModel:
    """A simple 3D CNN model for binary aneurysm classification."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 256)):
        """
        Initialize the model.
        
        Parameters
        ----------
            input_shape: Shape of input volumes (width, height, depth)
        """
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """
        Returns
        ----------
            Compiled Keras model
        """
        inputs = keras.Input((*self.input_shape, 1))
        
        x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding="same")(inputs)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(units=512, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(units=1, activation="sigmoid")(x)
        
        model = keras.Model(inputs, outputs, name="aneurysm_detection_3dcnn")
        return model
    
    def compile_model(self, learning_rate: float = 0.0001):
        """
        Compile the model with optimizer and loss.
        
        Parameters
        ----------
            learning_rate: Initial learning rate
        """
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True
        )
        
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )
    
    def summary(self):
        """Print model architecture summary."""
        self.model.summary()
    
    def get_model(self) -> keras.Model:
        """Return the Keras model."""
        return self.model