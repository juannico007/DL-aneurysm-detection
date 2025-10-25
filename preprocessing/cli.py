# preprocess/cli.py
from pathlib import Path
from .pipeline import Preprocess

def main():
    """Entry point for running preprocessing with the default configuration."""
    p = Preprocess(
        input_root=Path("mini-rsna-intracranial-aneurysm-detection"),
        output_root=Path("ct_preprocessed"),
        voxel_size=(1, 1, 1),
        shape=(256, 256, 256),
        output_format="npz",
        pipeline_version="0.0.1",
        oversub_factor=1.5,
    )
    p.run(batch_size=2)

if __name__ == "__main__":
    main()
