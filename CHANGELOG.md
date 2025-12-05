# Changelog
All notable changes to this project will be documented here. If needed, images with the model will be shown. 

## [0.0.6] - 2025-11-26

### Added
- Cloud training support with AWS SageMaker using Hugging Face Accelerate.
- Model versioning system based on GitHub username, model name, and version.
- Scripts to pull trained models from S3 and view SageMaker training logs from the command line
- `.env` configuration support for sensitive information like AWS credentials.

### Changed
- Data loading pipeline switched from multiple `.npz` files to a single HDF5 file
- Training entry point updated to `run_training.py` with `--mode` argument for local or cloud execution.
- New `AneurysmDataset` class for efficient data loading from HDF5 files.

### Fixed
- Accelerate configuration bug fixed via a patch script.

### Deprecated
- Old data loading methods using `.npz` files.

### Performance

| Configuration                   | Model Size | Parameters | Dataset       | Hardware                                             | Training Time per 1 epoch |
| ------------------------------- | ---------- | ---------- | ------------- | ---------------------------------------------------- | -------------- |
| Local | ~3 GB      | 328,001    | 777 CTA scans | RTX 4060 (8 GB VRAM), 24 GB RAM, Intel i9                            | **~3:45 minutes**  |
| Cloud   | ~3 GB      | 328,001     | 777 CTA scans | NVIDIA A10G (24 GB VRAM), 16 GB RAM, AMD EPYC 7R32 | **~1:45 minutes** |

## [0.0.3] - 2025-10-25

### Added
- `StreamingDataset` for on-demand data loading with async caching.
- Checkpointing system in preprocessing to resume interrupted runs.
- Kill switch for multi-threaded preprocessing (`Ctrl+C` support).
- Gradient accumulation to simulate larger batch sizes.
- GPU-based augmentation (`rotate_batch_gpu`) for faster transformations.
- Shared `accelerate_config.yaml` and `ds_config.json` for the optimization setup.

### Changed
- Preprocessing output format from `.nii.gz` (NIfTI) to `.npz` (NumPy compressed).
- All volumes converted from `float32` to `float16` for memory reduction.
- Removed sigmoid from model forward pass; use `BCEWithLogitsLoss` instead.
- Training optimized with Hugging Face Accelerate and 8-bit Adam optimizer.
- Datasets now pad dynamically to batch’s max size (multiple of 16).

### Fixed
- Crashes on missing or malformed CT series (improved error handling).
- Preprocessing hang on multithreaded shutdown (added safe thread termination).

### Deprecated
- Full in-memory dataset loading (replaced by `StreamingDataset`).

### Performance
- **9.6× smaller data** (from 19.2 GB → 2.01 GB).
- **>400× faster training** (from ~37 hours → ~5 minutes).


## [0.0.2] - 2025-xx-xx
### Fixed
- `ImageSeriesReader` retry logic for partial series directories.


## [0.0.1] - 2025-xx-xx
### Fixed
- `ImageSeriesReader` retry logic for partial series directories.
