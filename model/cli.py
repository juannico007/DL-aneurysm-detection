from pathlib import Path
from .data_loader import ScanDataLoader
from .model import AneurysmDetectionModel
from .training_pipeline import TrainingPipeline
import torch
import torch.distributed as dist
from torchsummary import summary
import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    """Entrypoint for preparing model and training with preprocessed images."""
    DATA_DIR = Path("ct_preprocessed/series")
    CSV_PATH = Path("ct_preprocessed/train.csv")
    INPUT_SHAPE = (256, 256, 256)
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION = 4 # simmulate: batch_size = batch_size * gradient_accumulation
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    TRAIN_RATIO = 0.7
    LEARNING_RATE_REDUCTION_EPOCHS = 4
    CACHE_SIZE = 8
    RADIUS = 15
    
    device = "cuda"

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    
    print("\n[1/5] Loading dataset...")
    data_loader = ScanDataLoader(DATA_DIR, CSV_PATH, proportion_to_use=0.15)
    x_train, y_train, x_val, y_val = data_loader.split_data(train_ratio=TRAIN_RATIO)
    
    print("\n[2/5] Building model...")
    model = AneurysmDetectionModel(input_shape=INPUT_SHAPE)
    if torch.cuda.is_available():
        model = model.to(device)
        print(summary(model, input_size = (1,256,256,256)))
    
    print("\n[3/5] Setting up training pipeline...")
    pipeline = TrainingPipeline(
        model=model,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_reduction_patience = LEARNING_RATE_REDUCTION_EPOCHS,
        cache_size=CACHE_SIZE,
        grad_accum_steps=GRADIENT_ACCUMULATION
    )
    
    print("\n[4/5] Creating Pytorch dataloaders...")
    train_dataset, val_dataset = pipeline.create_dataloaders(
        x_train, y_train, x_val, y_val
    )
    
    print("\n[5/5] Training model...")
    history = pipeline.train(train_dataset, val_dataset)
    
    print("Training completed!")
    print(f"Best model saved to: {pipeline.checkpoint_path}")

    with open('history.pickle', 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("history saved to history.pickle")

if __name__ == "__main__":
    main()
