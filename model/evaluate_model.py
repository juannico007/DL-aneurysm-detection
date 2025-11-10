from pathlib import Path
from .data_loader import ScanDataLoader
from .model import AneurysmDetectionModel
from .training_pipeline import TrainingPipeline
from accelerate import Accelerator
import torch
import torch.distributed as dist
from torchsummary import summary
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def safe_gather(tensor, device, accelerator):
    """Safely gather tensors across processes for Gloo backend."""
    tensor = tensor.flatten().contiguous()

    # Get max size across processes
    local_size = torch.tensor(tensor.shape[0], device=device)
    all_sizes = accelerator.gather(local_size)
    max_size = all_sizes.max().item()

    # Pad to max size
    current_size = tensor.shape[0]
    if current_size == max_size:
        return accelerator.gather(tensor)

    tensor = torch.nn.functional.pad(tensor, (0, max_size - current_size), value=0)

    return accelerator.gather(tensor)

def main():
    """Entrypoint for preparing model and training with preprocessed images."""
    DATA_DIR = Path("ct_preprocessed/series")
    CSV_PATH = Path("ct_preprocessed/train.csv")
    INPUT_SHAPE = (256, 256, 256)
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 4# simmulate: batch_size = batch_size * gradient_accumulation
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    TRAIN_RATIO = 0.7
    LEARNING_RATE_REDUCTION_EPOCHS = 4
    CACHE_SIZE = 32
    
    device = "cuda"

    accelerator = Accelerator(
            mixed_precision = "fp16", 
            gradient_accumulation_steps = GRADIENT_ACCUMULATION,
        )
    
    print("\n[1/5] Loading dataset...")
    data_loader = ScanDataLoader(DATA_DIR, CSV_PATH)
    x_train, y_train, x_val, y_val = data_loader.split_data(train_ratio=TRAIN_RATIO)
    
    print("\n[2/5] Loading model...")
    model = AneurysmDetectionModel()
    model.load_state_dict(torch.load("aneurysm_detection_best.pt", weights_only=True))
    print(type(model))
    if torch.cuda.is_available():
        model = model.to(device)
        print(summary(model, input_size = (1,256,256,256)))
    
    print("\n[3/5] Creating dataloaders...")
    pipeline = TrainingPipeline(
            model=model,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            lr_reduction_patience = LEARNING_RATE_REDUCTION_EPOCHS,
            cache_size=CACHE_SIZE,
            grad_accum_steps=GRADIENT_ACCUMULATION
        )

    train_dataloader, val_dataloader = pipeline.create_dataloaders(
        x_train, y_train, x_val, y_val
    )

    criterion = nn.BCEWithLogitsLoss()
    print("\n[4/5] Evaluating training set...")
    all_preds, all_labels = [], []
    for x_batch, y_batch in tqdm(train_dataloader):
        # y to float column vector
        y_batch = y_batch.float().unsqueeze(1)

        x_batch = x_batch.to(device, dtype=torch.float32, non_blocking=True)
        y_batch = y_batch.to(device, dtype=torch.float32, non_blocking=True)

        with accelerator.autocast():
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

        # gather predictions/labels only for metrics
        all_preds.append(outputs.detach())
        all_labels.append(y_batch.detach())

    all_preds = torch.cat(all_preds, dim=0).to(device, dtype=torch.float32).contiguous()
    all_labels = torch.cat(all_labels, dim=0).to(device, dtype=torch.float32).contiguous()

    # Now safe to gather on GPU
    all_preds  = safe_gather(all_preds, device, accelerator)
    all_labels = safe_gather(all_labels, device, accelerator)
    
    probs = torch.sigmoid(all_preds)
    preds = (probs >= 0.5).int().cpu().numpy()
    labs  = all_labels.int().cpu().numpy()

    train_acc = accuracy_score(labs, preds)
    train_conf_mat = confusion_matrix(labs, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=train_conf_mat,)
    disp.plot(cmap='Blues')  # optional: add color map
    plt.title("Training Confusion Matrix")

    # Save the figure
    plt.savefig("train_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()  # close the figure to free memory
    print("train acc", train_acc)
    
    print("\n[4/5] Evaluating validation set...")
    all_preds, all_labels = [], []
    for x_batch, y_batch in tqdm(val_dataloader):
        # y to float column vector
        y_batch = y_batch.float().unsqueeze(1)

        x_batch = x_batch.to(device, dtype=torch.float32, non_blocking=True)
        y_batch = y_batch.to(device, dtype=torch.float32, non_blocking=True)

        with accelerator.autocast():
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

        # gather predictions/labels only for metrics
        all_preds.append(outputs.detach())
        all_labels.append(y_batch.detach())

    all_preds = torch.cat(all_preds, dim=0).to(device, dtype=torch.float32).contiguous()
    all_labels = torch.cat(all_labels, dim=0).to(device, dtype=torch.float32).contiguous()
    # Now safe to gather on GPU
    all_preds  = safe_gather(all_preds, device, accelerator)
    all_labels = safe_gather(all_labels, device, accelerator)
    probs = torch.sigmoid(all_preds)
    preds = (probs >= 0.5).int().cpu().numpy()
    labs  = all_labels.int().cpu().numpy()

    val_acc = accuracy_score(labs, preds)
    val_conf_mat = confusion_matrix(labs, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=val_conf_mat,)
    disp.plot(cmap='Blues')  # optional: add color map
    plt.title("Valdiation Confusion Matrix")

    # Save the figure
    plt.savefig("val_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()  # close the figure to free memory
    print("val acc", val_acc)
    
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
