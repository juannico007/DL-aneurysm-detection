# Project Deep Learning

[![Version](https://img.shields.io/badge/version-v0.0.3-informational)](#)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![License](https://img.shields.io/badge/license-MIT-blue)](#)

- **v0.0.1** → Preprocessing (DICOM → NIfTI conversion)  
- **v0.0.2** → Baseline model definition and training  
- **v0.0.3** → Major optimization and scalability update

## ✨ What’s new in v0.0.3
- **NumPy compression:** Preprocessing now exports `.npz` files + float16 (1.75× smaller compared to the last version).
- **Preprocess checkpoint:** Continue preprocessing from where you left it last time. 
- **Asynchronous loading:** Introduced `StreamingDataset` for low-memory training.
- **Gradient accumulation:** Enables large effective batch sizes.
- **GPU augmentation:** Real-time data rotations during training.
- **Accelerate + DeepSpeed:** Optimized mixed-precision and distributed setup.

➡️ Full notes: see [CHANGELOG](./CHANGELOG.md#003---2025-10-25)  
➡️ Deep dive: [v0.0.3 release notes](./docs/release-notes/v0.0.3.md)  
➡️ Breaking changes? Check [MIGRATING.md](./MIGRATING.md)

## Quick Start

### 1. Install depndencies
```bash
pip install -r requirements.txt
```

### 2. Download the data

All the data: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection    
Mini set with 200 CTAs: https://www.kaggle.com/datasets/mihaibivol25/mini-rsna-intracranial-aneurysm-detectionn

### 3. Preprocess the data
```bash
python -m preprocessing.cli
```

### 4. Configure Accelerate
```bash
accelerate config
```
Make sure to be similar as in the `model/accelerate_config.yaml` file.

### 5. Launch Training
```bash
accelerate launch -m model.cli
```