from pathlib import Path
from tqdm import tqdm
import itk
from typing import Literal

class Preprocess:
    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        num_workers: int = 8,
        pipeline_version: str = "0.0.1",
        output_format: Literal["nii", "nii.gz"] = "nii.gz",
        voxel_size: tuple[float, float, float] = (1, 1, 1),
    ):
        for name, value in locals().items():
            if name != "self":
                setattr(self, name, value)
        
    
    def resample_image_and_convert_to_nii(self, user_id):
        """
        Resample a 3D image to the specified voxel size and convert to NIfTI format. Returns the resampled image (in itk format).
        """

        # Load your 3D image (from DICOM or already built)
        image = itk.imread(self.input_root / "series" / user_id, itk.F)  # itk.imread can read a folder of DICOMs too

        # Original info
        original_spacing = image.GetSpacing()
        original_size = image.GetLargestPossibleRegion().GetSize()
        original_origin = image.GetOrigin()
        original_direction = image.GetDirection()
        
        # Compute new size to keep the same physical extent
        new_size = [
            int(round(orig_sz * orig_spc / new_spc))
            for orig_sz, orig_spc, new_spc in zip(original_size, original_spacing, self.voxel_size)
        ]
        
        # Resampling
        resample = itk.ResampleImageFilter.New(Input=image)
        resample.SetInterpolator(itk.LinearInterpolateImageFunction.New(InputImage=image))
        resample.SetOutputSpacing(self.voxel_size)
        resample.SetSize(new_size)
        resample.SetOutputOrigin(original_origin)
        resample.SetOutputDirection(original_direction)
        resample.SetTransform(itk.IdentityTransform[itk.D, 3].New())
        
        resample.Update()
        resampled_image = resample.GetOutput()
        
        return resampled_image

    def preprocess_generate_metadata(self):
        """
        Generate metadata about the dataset:
        - Count number of patients
        - Count number of series
        - Count modalities (CTA, MRA, DSA, etc)
        - Count number of segmentations available
        Save this metadata as a JSON file in the output_root.
        """
        import json
        import pandas as pd

        train_csv = self.input_root / "train.csv"
        df = pd.read_csv(train_csv)

        num_series = df["SeriesInstanceUID"].nunique()
        modality_counts = df["Modality"].value_counts().to_dict()

        metadata = {
            "num_series": num_series,
            "modality_counts": modality_counts,
            "pipeline_version": self.pipeline_version,
            "voxel_size": self.voxel_size,
        }

        output_metadata_path = self.output_root / "metadata.json"
        self.output_root.mkdir(parents=True, exist_ok=True)
        with open(output_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def run(self):
        """
        Main function to run the preprocessing:
        - Generate metadata
        - For each series in input_root/series:
            * Resample and convert to NIfTI
            * Save to output_root with same folder structure
        """
        self.preprocess_generate_metadata()

        series_dir = self.input_root / "series"
        output_series_dir = self.output_root / "series"
        output_series_dir.mkdir(parents=True, exist_ok=True)

        series_ids = [d.name for d in series_dir.iterdir() if d.is_dir()]

        for series_id in tqdm(series_ids, desc="Processing series"):
            resampled_image = self.resample_image_and_convert_to_nii(series_id)
            output_path = output_series_dir / f"{series_id}.{self.output_format}"
            itk.imwrite(resampled_image, str(output_path))


Preprocess(
    input_root=Path("../ct_subset"),
    output_root=Path("../ct_preprocessed"),
    voxel_size=(1, 1, 1),
    output_format="nii.gz",
    num_workers=8,
    pipeline_version="0.0.1"
).run()