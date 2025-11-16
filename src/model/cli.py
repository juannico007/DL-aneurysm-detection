from pathlib import Path
from torchsummary import summary
import pickle
import torch
import os
import torch.distributed as dist
import json
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODE_ENV_VAR = "TRAINING_MODE"

HYPERPARAMS = {
    "data_dir": "h5-aneurysm",
    "csv_path": "train.csv",
    "input_shape": (256, 256, 256),
    "batch_size": 2,
    "grad_accum": 4,
    "epochs": 5,
    "learning_rate": 1e-4,
    "train_ratio": 0.7,
    "lr_reduction_epochs": 4,
}

ENV_FILE = Path(".env")


def load_env_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Cloud mode enabled but env file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


def maybe_init_dist():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size == 1:
        return  # nothing to do

    if dist.is_initialized():
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    # force gloo when not on SageMaker
    if "SM_MODEL_DIR" not in os.environ:
        backend = "gloo"

    dist.init_process_group(backend=backend)


def main():
    hyperparams = HYPERPARAMS.copy()
    mode = os.environ.get(MODE_ENV_VAR, hyperparams.get("mode", "local"))
    if mode not in {"local", "cloud"}:
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'local' or 'cloud'.")
    if mode == "local" and ("SM_MODEL_DIR" in os.environ or "SM_OUTPUT_DATA_DIR" in os.environ):
        mode = "cloud"
    hyperparams["mode"] = mode

    if mode == "cloud":
        load_env_file(ENV_FILE)
    elif "H5_PATH" not in os.environ and ENV_FILE.exists():
        load_env_file(ENV_FILE)

    github_user = os.environ.get("GITHUB_USER")
    model_name = os.environ.get("RUN_MODEL_NAME")
    model_version = os.environ.get("RUN_MODEL_VERSION")

    # maybe_init_dist()

    # Heavy imports AFTER deps are ensured

    from .data_loader import ScanDataLoader
    from .model import AneurysmDetectionModel
    from .training_pipeline import TrainingPipeline

    input_shape = tuple(hyperparams["input_shape"])

    # SageMaker model / output directories (or "." locally)
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "."))
    output_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "."))

    using_sagemaker = mode == "cloud"
    artifact_id = None
    if using_sagemaker:
        missing = [
            name
            for name, value in [
                ("GITHUB_USER", github_user),
                ("RUN_MODEL_NAME", model_name),
                ("RUN_MODEL_VERSION", model_version),
            ]
            if not value
        ]
        if missing:
            raise ValueError(
                f"SageMaker runs require GitHub tagging. Missing env vars: {', '.join(missing)}"
            )
        artifact_id = f"{github_user}_{model_name}_{model_version}"

    model_artifact_dir = model_dir / artifact_id if artifact_id else model_dir
    output_artifact_dir = output_dir / artifact_id if artifact_id else output_dir
    model_artifact_dir.mkdir(parents=True, exist_ok=True)
    output_artifact_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = f"{artifact_id}.pt" if artifact_id else "aneurysm_detection_best.pt"
    checkpoint_path = model_artifact_dir / checkpoint_name
    history_path = output_artifact_dir / "history.pickle"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def resolve_h5_path(preferred: Path) -> Path:
        env_value = os.environ.get("H5_PATH")
        if env_value:
            path = Path(env_value).expanduser()
            if path.is_absolute():
                return path
            return Path(".") / path

        train_channel = os.environ.get("SM_CHANNEL_TRAIN")
        preferred = Path(preferred)
        if preferred.is_absolute():
            return preferred
        if train_channel:
            return Path(train_channel) / preferred
        return Path(".") / preferred

    h5_path = resolve_h5_path(Path(hyperparams["data_dir"]))

    print("\n[1/5] Loading dataset...")
    data_loader = ScanDataLoader(
        h5_path=h5_path,
        csv_path=Path(hyperparams["csv_path"]),
    )
    resolved_h5_path = Path(data_loader.h5_path)
    x_train, y_train, x_val, y_val = data_loader.split_data(train_ratio=hyperparams["train_ratio"])

    print("\n[2/5] Building model...")
    model = AneurysmDetectionModel(input_shape=input_shape)
    if torch.cuda.is_available():
        model = model.to(device)
        print(summary(model, input_size=(1,) + input_shape))

    print("\n[3/5] Setting up training pipeline...")
    pipeline = TrainingPipeline(
        model=model,
        batch_size=hyperparams["batch_size"],
        epochs=hyperparams["epochs"],
        learning_rate=hyperparams["learning_rate"],
        lr_reduction_patience=hyperparams["lr_reduction_epochs"],
        grad_accum_steps=hyperparams["grad_accum"],
        checkpoint_path=str(checkpoint_path),
    )

    print("\n[4/5] Creating Pytorch dataloaders...")
    train_dataset, val_dataset = pipeline.create_dataloaders(
        h5_path=resolved_h5_path,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )

    print("\n[5/5] Training model...")
    history = pipeline.train()

    print("Training completed!")
    print(f"Best model saved to: {pipeline.checkpoint_path}")

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "wb") as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"History saved to {history_path}")

    hyperparams_path = output_artifact_dir / "hyperparameters.json"
    with open(hyperparams_path, "w", encoding="utf-8") as handle:
        json.dump(hyperparams, handle, indent=2, sort_keys=True)
    print(f"Hyperparameters saved to {hyperparams_path}")

    if artifact_id:
        metadata = {
            "artifact_id": artifact_id,
            "github_user": github_user,
            "model_name": model_name,
            "model_version": model_version,
            "training_job_name": os.environ.get("TRAINING_JOB_NAME", "sagemaker"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "files": {
                "checkpoint": checkpoint_path.name,
                "history": history_path.name,
                "hyperparameters": hyperparams_path.name,
            },
            "cli_args": hyperparams,
        }
        metadata_path = output_artifact_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
