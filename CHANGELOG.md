# Changelog
All notable changes to this project will be documented here. If needed, images with the model will be shown. 

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
- Added `cache_size` parameter to limit dataset memory usage.
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
