# Preprocessing Documentation

This document provides an overview of the preprocessing steps for the dataset as well as how to add new functions to the preprocessing pipeline.

## Overview of Preprocessing Steps

The main steps in the preprocessing pipeline include:
1. **Resample Images**: Standardizing the voxel size across all images. (For now is (1,1,1))
2. **Pre Hooks**: Custom functions that can be applied to the images before the main preprocessing steps. (For example skull stripping, or bias field correction)
3. **Crop Images**: Reducing the image size to focus on the brain (it can be different depending on the modality (CT or MRI)).
4. **Normalize Images**: Adjusting the intensity values of the images to a standard range. (Can be also modality dependent min-max normalization for CT and z-score normalization for MRI)
5. **Post Hooks**: Custom functions that can be applied to the images after the main
(example: histogram equalization, and )

There will be created a `metadata file` which includes the settings used for preprocessing the dataset.

Create a `preprocess overview` to check the quality of the preprocessing steps at the end. The generated `preprocessing_overview.png` includes an axial slice, intensity histogram, and summary statistics for a representative series before and after preprocessing.

## Adding New Functions to the Pipeline

In policies folder you can create a new file for the specific modality. For example, if you want to add a new preprocessing step for MRI images, you can create a file named `mri.py` and implement the desired functions. You need to use the @register("[name_of_the_modality]") decorator, create a class that inherits from ModalityPolicy, and implement the functions you want to change. If you have a function for normalization, you can implement in the normalize.py, and so on. 

## Explanation of the Code Structure
The preprocessing pipeline is structured into several key components:
- **pipeline.py**: This file contains the main preprocessing pipeline class, `Preprocess`, which orchestrates the entire preprocessing workflow. It handles loading images, applying the preprocessing steps, and saving the processed images.
- **policy.py**: This file defines the `ModalityPolicy` class, which serves as a base class for modality-specific preprocessing policies. It includes methods for each preprocessing step that can be overridden in subclasses.
- **policies/**: This directory contains modality-specific policy implementations. Each file in this directory defines a class that inherits from `ModalityPolicy` and implements the preprocessing steps for a specific imaging modality (e.g., CT, MRI).
- **cli.py**: This file provides a command-line interface for running the preprocessing pipeline. It allows users to specify input and output directories, voxel size, and other parameters.
- **register.py**: This file contains the `register` decorator, which is used to register modality-specific policies. It maintains a registry of available policies that can be selected based on the imaging modality.
- **workers.py**: This file defines worker classes that handle parallel processing of images. It includes the `PreprocessWorker` class, which processes individual images using the specified preprocessing policy.

## Example Usage
To run the preprocessing pipeline, you can use the command-line interface provided in `cli.py`. Here is an example of how to execute the preprocessing with default settings:

```bash
python3 -m preprocessing.cli
```

## Experiements
- First we created the preprocessing pipeline with the basic steps (resampling, cropping, normalizing). It took ~40 seconds per series (RTX 4060 8GB vram, ultra i9, 24 GB RAM).
- Then we parallelized the preprocessing, which reduced the time to ~20 seconds per series.
- We added batch processing to allow multiple series to be processed simultaneously on one core, and also added a thread pool to each core, while also giving more threads to heavy series.

For example, we can have a batch size of 3, with 5 workers (processes), and each worker can have 6 threads. This means that at most 15 series can be processed simultaneously, with each series using up to 6 threads for processing. This configuration can significantly speed up the preprocessing time, especially for large datasets.
