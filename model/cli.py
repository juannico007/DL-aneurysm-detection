from pathlib import Path
from .data_loader import ScanDataLoader
from .model import AneurysmDetectionModel
from .training_pipeline import TrainingPipeline

def main():
    """Entrypoint for preparing model and training with preprocessed images."""
    DATA_DIR = Path("ct_preprocessed/series")
    CSV_PATH = Path("ct_subset/train.csv")
    INPUT_SHAPE = (256, 256, 256)
    BATCH_SIZE = 2
    EPOCHS = 100
    LEARNING_RATE = 0.0001
    TRAIN_RATIO = 0.7
    
    print("\n[1/5] Loading dataset...")
    data_loader = ScanDataLoader(DATA_DIR, CSV_PATH)
    x_train, y_train, x_val, y_val = data_loader.split_data(train_ratio=TRAIN_RATIO)
    
    print("\n[2/5] Building model...")
    model = AneurysmDetectionModel(input_shape=INPUT_SHAPE)
    model.compile_model(learning_rate=LEARNING_RATE)
    model.summary()
    
    print("\n[3/5] Setting up training pipeline...")
    pipeline = TrainingPipeline(
        model=model,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    
    print("\n[4/5] Creating TensorFlow datasets...")
    train_dataset, val_dataset = pipeline.create_datasets(
        x_train, y_train, x_val, y_val
    )
    
    print("\n[5/5] Training model...")
    history = pipeline.train(train_dataset, val_dataset)
    
    print("Training completed!")
    print(f"Best model saved to: {pipeline.checkpoint_path}")


if __name__ == "__main__":
    main()
