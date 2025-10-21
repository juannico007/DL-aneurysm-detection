# preprocess/cli.py
from pathlib import Path
from .pipeline import Preprocess

def main():
    """Entry point for running preprocessing with the default configuration."""
    p = Preprocess(
        input_root=Path("preprocessing_failing"),
        output_root=Path("preprocessed_not_failing"),
        voxel_size=(1, 1, 1),
        shape=(256, 256, 256),
        output_format="nii.gz",
        pipeline_version="0.0.1",
        oversub_factor=1.5,
    )
    p.run(batch_size=1)

if __name__ == "__main__":
    main()
