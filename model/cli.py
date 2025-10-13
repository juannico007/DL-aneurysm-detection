from pathlib import Path
import itk
import os
import pandas as pd
import numpy as np
import random
from scipy import ndimage
import tensorflow as tf
import keras
from keras import layers

def process_scan(path):
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    cache_path = cache_dir / (Path(path).stem + ".npy")
    if cache_path.exists():
        return np.load(cache_path)
    
    image = itk.imread(path, itk.F)
    volume = itk.GetArrayFromImage(image).astype(np.float32)
    print(f"Processed volume {path}")

    np.save(cache_path, volume)
    return volume

def rotate(volume):
    def scipy_rotate(volume):
        angles = [-20, -10, -5, 5, 10, 20]
        angle = random.choice(angles)
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=-1)
    return volume, label


def validation_preprocessing(volume, label):
    volume = tf.expand_dims(volume, axis=-1)
    return volume, label

def get_model(width=128, height=128, depth=64):
    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


def main():
    data_dir = Path("ct_preprocessed/series")
    csv_path = Path("ct_subset/train.csv")
    
    df = pd.read_csv(csv_path)

    image_paths = []
    labels = []

    for uid, label in zip(df["SeriesInstanceUID"], df["Aneurysm Present"].values):
        path = os.path.join(data_dir, f"{uid}.nii.gz")
        if os.path.exists(path):
            image_paths.append(path)
            labels.append(label)
        else:
            print(f"Missing file: {path}")

    abnormal_paths = [p for p, l in zip(image_paths, labels) if l == 1]
    normal_paths   = [p for p, l in zip(image_paths, labels) if l == 0]

    abnormal_scans = np.array([process_scan(p) for p in abnormal_paths])
    normal_scans = np.array([process_scan(p) for p in normal_paths])

    abnormal_labels = np.ones(len(abnormal_scans))
    normal_labels = np.zeros(len(normal_scans))
    print(f"Sorted Labels")

    split_idx_abn = int(0.7 * len(abnormal_labels))
    split_idx_norm = int(0.7 * len(normal_scans))

    x_train = np.concatenate((abnormal_scans[:split_idx_abn], normal_scans[:split_idx_norm]), axis=0)
    y_train = np.concatenate((abnormal_labels[:split_idx_abn], normal_labels[:split_idx_norm]), axis=0)
    print(f"Prepared Train Set")

    x_val = np.concatenate((abnormal_scans[split_idx_abn:], normal_scans[split_idx_norm:]), axis=0)
    y_val = np.concatenate((abnormal_labels[split_idx_abn:], normal_labels[split_idx_norm:]), axis=0)
    print(f"Prepared Validation Set")

    print(f"Train samples: {x_train.shape[0]}, Validation samples: {x_val.shape[0]}")

    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    print(f"Prepared loaders")

    batch_size = 2
    train_dataset = (
        train_loader.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )
    print(f"Prepared train dataset")

    validation_dataset = (
        validation_loader.shuffle(len(x_val))
        .map(validation_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )
    print(f"Prepared validation dataset")

    model = get_model(width=256, height=256, depth=256)

    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer= tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
        run_eagerly=True,
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        "3d_image_classification.keras", save_best_only=True
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

    # Train the model
    epochs = 100
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )


if __name__ == "__main__":
    main()