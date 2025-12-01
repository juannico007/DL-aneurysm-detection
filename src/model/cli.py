from pathlib import Path
from torchsummary import summary
import pickle
import torch
import os
import torch.distributed as dist
import json
from datetime import datetime
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODE_ENV_VAR = "TRAINING_MODE"

HYPERPARAMS = {
    "data_dir": "train_dataset.h5",
    "patch_csv": "train_patches.csv",
    "proportion_to_use": 1,
    "input_shape": (256, 256, 256),
    "batch_size": 20,
    "grad_accum": 2,
    "epochs": 40,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    # "scheduler": {
    "max_voxels": 500 * 500 * 500,
    "loss_weights": {
        "alpha_max": 0.3,
        "beta": 1.0,
        "lambda_tail": 0.1,
        "lambda_max": 0.1,
        "background": 0.1,
        "neg_ramp_epochs": 20,
        "alpha_min": 0.05,
        "alpha_schedule": "cosine",
    },
    "neg_warmup_epochs": 0,
    "heatmap_sigma": 5,
    # Heatmap decay strategy: {"name": "epoch"|"plateau"|"threshold", ...}
    "heatmap_decay": {
        "min_sigma": 3.0,
        "name": "plateau",
        # "epoch": 10,          # used when name == "epoch"
        # For plateau mode:
        "metric": "val_loss",
        "mode": "min",      # or "max"
        "patience": 10,
        "factor": 0.9,
        # For threshold mode:
        # "metric": "val_peak_err",
        # "mode": "min",
        # "threshold": 12.0,
        # "factor": 0.8,
    },
    "scheduler":{
        "name" : "cosine",
        "eta_min": 1e-6
    },

    # "scheduler": {
    #     "name": "plateau",
    #     "mode": "min",
    #     "factor": 0.5,
    #     "patience": 4,
    #     "min_lr": 1e-6,
    # },

    # "scheduler": {
    #     "name": "onecycle",
    # },
    
    "train_ratio": 0.7,
    "lr_reduction_epochs": 10,
    "radius": 5,
    "unet": {
        "in_channels": 1,
        "out_channels": 1,
        "n_blocks": 4,
        "start_filters": 16,
        "activation": "relu",
        "normalization": "group4",
        "conv_mode": "same",
        "up_mode": "trilinear",
        "middle_neurons": 256,
        "class_output": 1,
        "dropout": 0.2,
        "attention": True,
        "regression": True,
    },
}

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


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


def resolve_local_cloud_dir(github_user: str, model_name: str, model_version: str) -> Path:
    """
    Returns cloud_models/<user>/<model_version> under the repo root unless LOCAL_CLOUD_ROOT overrides it.
    """
    base_override = os.environ.get("LOCAL_CLOUD_ROOT")
    if base_override:
        base = Path(base_override).expanduser()
        if not base.is_absolute():
            base = REPO_ROOT / base
    else:
        base = REPO_ROOT / "cloud_models"
    return base / github_user / f"{model_name}_{model_version}"


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

    # Code for choosing the mode (local vs cloud/SageMaker)
    mode = os.environ.get(MODE_ENV_VAR, hyperparams.get("mode", "local"))
    if mode not in {"local", "cloud"}:
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'local' or 'cloud'.")
    if mode == "local" and ("SM_MODEL_DIR" in os.environ or "SM_OUTPUT_DATA_DIR" in os.environ):
        mode = "cloud"
    hyperparams["mode"] = mode

    if mode == "cloud":
        load_env_file(ENV_FILE)
    elif ENV_FILE.exists():
        load_env_file(ENV_FILE)

    github_user = os.environ.get("GITHUB_USER")
    model_name = os.environ.get("RUN_MODEL_NAME")
    model_version = os.environ.get("RUN_MODEL_VERSION")

    # maybe_init_dist()

    # Heavy imports AFTER deps are ensured

    from .unet import UNet
    from .training_pipeline import TrainingPipeline, load_patch_records

    input_shape = tuple(hyperparams["input_shape"])

    # SageMaker model / output directories (or "." locally)
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "."))
    output_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "."))

    using_sagemaker = mode == "cloud"
    artifact_id = None
    nest_artifact_subdir = False
    if using_sagemaker: # save artifacts in cloud mode
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
        nest_artifact_subdir = True
    else: # save artifacts in local mode
        local_missing = [
            name
            for name, value in [
                ("GITHUB_USER", github_user),
                ("RUN_MODEL_NAME", model_name),
                ("RUN_MODEL_VERSION", model_version),
            ]
            if not value
        ]
        if local_missing:
            print(
                "Local run: cloud_models export disabled because these env vars are missing: "
                + ", ".join(local_missing)
            )
        else:
            artifact_id = f"{github_user}_{model_name}_{model_version}"
            local_artifact_root = resolve_local_cloud_dir(github_user, model_name, model_version)
            model_dir = local_artifact_root
            output_dir = local_artifact_root
            print(f"Local artifacts will be stored in {local_artifact_root}")

    # artifact directories
    model_artifact_dir = model_dir / artifact_id if artifact_id and nest_artifact_subdir else model_dir
    output_artifact_dir = output_dir / artifact_id if artifact_id and nest_artifact_subdir else output_dir
    model_artifact_dir.mkdir(parents=True, exist_ok=True)
    output_artifact_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = f"{artifact_id}.pt" if artifact_id else "aneurysm_detection_best.pt"
    checkpoint_path = model_artifact_dir / checkpoint_name
    history_path = output_artifact_dir / "history.pickle"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get the h5 path both for local and cloud runs
    def resolve_h5_path(preferred: Path) -> Path:
        train_channel = os.environ.get("SM_CHANNEL_TRAIN")
        preferred = Path(preferred)
        if preferred.is_absolute():
            return preferred
        if train_channel:
            return Path(train_channel) / preferred
        return Path(".") / preferred

    h5_path = resolve_h5_path(Path(hyperparams["data_dir"]))
    patch_csv = Path(hyperparams["patch_csv"])
    proportion = float(hyperparams.get("proportion_to_use", 1.0))
    split_seed = int(hyperparams.get("split_seed", 42))

    print("\n[1/5] Loading patch metadata...")
    records = load_patch_records(patch_csv)
    if not records:
        raise ValueError(f"No records found in {patch_csv}")

    # patient-level subsampling by proportion_to_use
    series_ids = sorted({r.series_id for r in records})
    if not (0 < proportion <= 1):
        raise ValueError("proportion_to_use must be in (0,1].")
    if proportion < 1.0:
        rng = np.random.default_rng(split_seed)
        keep_count = max(1, int(round(len(series_ids) * proportion)))
        keep_series = set(rng.choice(series_ids, size=keep_count, replace=False))
        records = [r for r in records if r.series_id in keep_series]
        print(f"[Info] Using {keep_count}/{len(series_ids)} series (~{proportion*100:.1f}%).")
    else:
        keep_series = set(series_ids)

    # report counts
    pos_count = sum(1 for r in records if r.label == 1)
    neg_count = sum(1 for r in records if r.label == 0)
    print(f"[Info] Patch counts -> positives: {pos_count}, negatives: {neg_count}, total: {len(records)}")
    print(f"[Info] Series covered: {len({r.series_id for r in records})}")

    resolved_h5_path = Path(h5_path)

    print("\n[2/5] Building model...")

    unet_kwargs = dict(hyperparams.get("unet", {}))
    model = UNet(**unet_kwargs)

    if torch.cuda.is_available():
    #     # Run the summary on CPU to avoid exhausting limited GPU VRAM
    #     print(summary(model.to("cpu"), input_size=(1,) + input_shape, device="cpu"))
         model = model.to(device)
    # else:
    #     print(summary(model, input_size=(1,) + input_shape, device="cpu"))

    print("\n[3/5] Setting up training pipeline...")
    pipeline = TrainingPipeline(
        model=model,
        batch_size=hyperparams["batch_size"],
        epochs=hyperparams["epochs"],
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
        lr_reduction_patience=hyperparams["lr_reduction_epochs"],
        grad_accum_steps=hyperparams["grad_accum"],
        checkpoint_path=str(checkpoint_path),
        radius=hyperparams["radius"],
        patch_csv=patch_csv,
        heatmap_sigma=hyperparams["heatmap_sigma"],
        heatmap_decay=hyperparams.get("heatmap_decay"),
        patch_size=hyperparams.get("patch_size", 64),
        train_ratio=hyperparams.get("train_ratio", 0.8),
        split_seed=hyperparams.get("split_seed", 42),
        scheduler_config=hyperparams["scheduler"],
        loss_weights=hyperparams.get("loss_weights"),
        neg_warmup_epochs=hyperparams.get("neg_warmup_epochs", 0),
    )

    print("\n[4/5] Creating Pytorch dataloaders...")
    train_dataset, val_dataset = pipeline.create_dataloaders(
        h5_path=resolved_h5_path,
        patch_csv=patch_csv,
        train_ratio=hyperparams.get("train_ratio", 0.8),
        split_seed=hyperparams.get("split_seed", 42),
        records=records,
    )

    print("\n[5/5] Training model...")
    history = pipeline.train()

    print("Training completed!")
    print(f"Best model saved to: {pipeline.checkpoint_path}")

    # save history
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "wb") as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"History saved to {history_path}")

    # save hyperparameters
    hyperparams_path = output_artifact_dir / "hyperparameters.json"
    with open(hyperparams_path, "w", encoding="utf-8") as handle:
        json.dump(hyperparams, handle, indent=2, sort_keys=True)
    print(f"Hyperparameters saved to {hyperparams_path}")

    # save metadata
    if artifact_id:
        metadata = {
            "artifact_id": artifact_id,
            "github_user": github_user,
            "model_name": model_name,
            "model_version": model_version,
            "training_job_name": os.environ.get(
                "TRAINING_JOB_NAME", "sagemaker" if using_sagemaker else "local-run"
            ),
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
