# Project Deep Learning

[![Version](https://img.shields.io/badge/version-v0.0.6-informational)](#)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![License](https://img.shields.io/badge/license-MIT-blue)](#)

- **v0.0.6** → Move the training to Cloud (SageMaker) + create a versioning system for models
- **v0.0.5** → Update Preprocessing to get aneurysms coordinates + sphere mask generator
- **v0.0.4** → Implement U-net architecture
- **v0.0.3** → Major optimization and scalability update
- **v0.0.2** → Baseline model definition and training  
- **v0.0.1** → Preprocessing (DICOM → NIfTI conversion)  

## ✨ What’s new in v0.0.6
- **Cloud training with SageMaker**: Launch training jobs directly on AWS SageMaker from your local machine using the `--mode cloud` flag. The training script automatically picks up configuration from the `.env` file for seamless integration.
- **Model versioning**: Each training run can now be tagged with a model name and version, making it easier to track and manage different iterations of your models.
- **Pull models**: A new script `scripts/pull_models.py` allows you to download trained models and checkpoints from S3 buckets based on GitHub user, model name, and version.
- **Show logs** : View real-time logs of your SageMaker training jobs directly from the command line using the `scripts/show_sagemaker_logs.py` script.

➡️ Full notes: see [CHANGELOG](./CHANGELOG.md#006---2025-11-26)  
➡️ Deep dive: [v0.0.6 release notes](./docs/release-notes/v0.0.6.md)  
➡️ Breaking changes? Check [MIGRATING.md](./MIGRATING.md)


## 📢 Relesae v0.0.5 - Preprocessing Update + Sphere Mask Generator

## 📢 Relesae v0.0.4 - Implement U-net Architecture

## 📢 Release v0.0.3 - Major Optimization Updat
- **NumPy compression:** Preprocessing now exports `.npz` files + float16 (1.75× smaller compared to the last version).
- **Preprocess checkpoint:** Continue preprocessing from where you left it last time. 
- **Asynchronous loading:** Introduced `StreamingDataset` for low-memory training.
- **Gradient accumulation:** Enables large effective batch sizes.
- **GPU augmentation:** Real-time data rotations during training.
- **Accelerate + DeepSpeed:** Optimized mixed-precision and distributed setup.

➡️ Full notes: see [CHANGELOG](./CHANGELOG.md#003---2025-10-25)  
➡️ Deep dive: [v0.0.3 release notes](./docs/release-notes/v0.0.3.md)  

## Quick Start

### 1. Install depndencies
```bash
pip install -r requirements.txt
```

### 2. Download the data
Download the h5 file and train.csv file from Kaggle:
All CTA data: https://www.kaggle.com/datasets/juannicolasquintero/rsna-intracranial-aneurysm-detection-cta/data?select=dataset.h5   

### 3. Launch Training
```bash
python src/run_training.py --mode local
```

Cloud (SageMaker) run (loads `src/.env` automatically):

```bash
python src/run_training.py --mode cloud
```