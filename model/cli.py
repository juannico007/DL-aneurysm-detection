from pathlib import Path
import itk
import os
import pandas as pd
import numpy as np
import random
from scipy import ndimage
import tensorflow as tf
from .model import AneurysmDetectionModel
from .data_augmentation import DataAugmentation

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
        .map(DataAugmentation.augment_training, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(2)
    )
    print(f"Prepared train dataset")

    validation_dataset = (
        validation_loader.shuffle(len(x_val))
        .map(DataAugmentation.prepare_validation, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(2)
    )
    print(f"Prepared validation dataset")

    initial_learning_rate = 0.0001
    model = AneurysmDetectionModel(input_shape=(256, 256, 256))
    model.compile_model(learning_rate=initial_learning_rate)
    model.summary()

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