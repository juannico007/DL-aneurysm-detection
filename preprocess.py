from pathlib import Path
from tqdm import tqdm
import itk
import numpy as np
from typing import Literal

class Preprocess:
    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        num_workers: int = 1,
        pipeline_version: str = "0.0.1",
        output_format: Literal["nii", "nii.gz"] = "nii.gz",
        voxel_size: tuple[float, float, float] = (1, 1, 1),
    ):
        for name, value in locals().items():
            if name != "self":
                setattr(self, name, value)

    def resample_image_and_convert_to_nii(self, user_id) -> itk.Image:
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

    def normalize_image(self, image: itk.Image, method: Literal["zscore", "minmax"] = "zscore") -> itk.Image:
        """
        Normalize image intensities using z-score or min-max scaling.
        
        Parameters
        ----------
        image : itk.Image
            Input image (after resampling).
        method : str, optional
            Normalization method ("zscore" or "minmax"). Default is "zscore".     
            minmax good for CT (standardized range); zscore good for MRI (varied range).
        
        Returns
        -------
        itk.Image
            Normalized image in the same itk.Image format.
        """

        # Convert ITK image to NumPy for manipulation
        array = itk.GetArrayFromImage(image).astype(np.float32)

        if method == "zscore":
            mean = np.mean(array)
            std = np.std(array)
            if std > 0:  # avoid divide-by-zero
                array = (array - mean) / std
            else:
                array = array - mean  # all pixels same → make them 0

        elif method == "minmax":
            min_val = np.min(array)
            max_val = np.max(array)
            if max_val > min_val:  # avoid divide-by-zero
                array = (array - min_val) / (max_val - min_val)
            else:
                array = array - min_val  # all pixels same → make them 0

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Convert back to ITK image with same metadata
        normalized_image = itk.GetImageFromArray(array)
        normalized_image.CopyInformation(image)

        return normalized_image


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


    def crop_head_from_ct(
    self,
    image: itk.Image,
    hu_air_threshold: float = -300.0,
    neck_band: tuple[float, float] = (0.2, 0.8),
    margin: int = 10,
) -> itk.Image:
        """
        Heuristic head-only crop for CT:
        1) Threshold > hu_air_threshold to get tissue mask.
        2) Along z, compute area per slice and pick the minimum within neck_band of the occupied span.
        3) Crop z from (neck_idx - margin) to the top, then tighten XY to nonzero mask within that z-range.

        Parameters
        ----------
        image : itk.Image
            Input (normalized) CT image.
        hu_air_threshold : float
            Threshold to exclude air/background (default: -300 HU).
        neck_band : (float, float)
            Fractional window (within occupied z-span) to search for neck minimum.
        margin : int
            Extra slices to keep below the neck.

        Returns
        -------
        itk.Image
            Head-cropped ITK image. Falls back to simple nonzero bbox if heuristic fails.
        """
        arr = itk.GetArrayFromImage(image)  # z, y, x
        arr = arr.astype(np.float32)

        # 1) Tissue mask (exclude air)
        tissue = arr > hu_air_threshold

        # If the mask is too small, bail out to original
        if tissue.sum() < 1000:
            return image

        # 2) Area per slice across z
        area = tissue.reshape(tissue.shape[0], -1).sum(axis=1)
        nz = np.where(area > 0)[0]
        if len(nz) < 10:
            return image  # not enough occupied slices to do a neck split

        z_lo, z_hi = nz[0], nz[-1]
        band_lo = z_lo + int((z_hi - z_lo) * neck_band[0])
        band_hi = z_lo + int((z_hi - z_lo) * neck_band[1])
        if band_hi <= band_lo:
            band_lo, band_hi = z_lo, z_hi

        band = area[band_lo:band_hi]
        if band.size == 0:
            return image

        # Neck index is the minimum area within the band (narrowest cross-section)
        neck_rel = int(np.argmin(band))
        neck_idx = band_lo + neck_rel

        # 3) Crop z from (neck - margin) up to the top occupied z
        z_start = max(neck_idx - margin, 0)
        z_end = z_hi + 1  # inclusive end -> +1 for slicing

        submask = tissue[z_start:z_end]
        if submask.sum() == 0:
            return image  # nothing above neck; bail out

        # Tighten XY to nonzero within the chosen z-range
        coords = np.array(np.nonzero(submask))
        zmin, ymin, xmin = coords.min(axis=1)
        zmax, ymax, xmax = coords.max(axis=1) + 1  # +1 for slicing

        cropped = arr[
            z_start + zmin : z_start + zmax,
            ymin:ymax,
            xmin:xmax,
        ]

        out = itk.GetImageFromArray(cropped)
        # Rebuild spatial metadata correctly
        # Compute new origin after cropping: index->physical mapping
        # Note: array indexing is in z,y,x; ITK index order is (x,y,z)
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()

        # Convert (z,y,x) offsets to physical shift
        offset_index = np.array([xmin, ymin, z_start + zmin], dtype=float)  # (x,y,z) in index space
        # Physical shift = direction * (offset_index * spacing)
        spacing_vec = np.array(spacing)
        dir_mat = np.array(direction).reshape(3, 3)
        physical_shift = dir_mat @ (offset_index * spacing_vec)

        new_origin = tuple(np.array(origin) + physical_shift)

        out.SetSpacing(spacing)
        out.SetDirection(direction)
        out.SetOrigin(new_origin)

        return out

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
            normalized_image = self.normalize_image(resampled_image, method="minmax")
            cropped_image = self.crop_head_from_ct(normalized_image)
            output_path = output_series_dir / f"{series_id}.{self.output_format}"
            itk.imwrite(cropped_image, str(output_path))

Preprocess(
    input_root=Path("ct_subset"),
    output_root=Path("ct_preprocessed"),
    voxel_size=(1, 1, 1),
    output_format="nii.gz",
    num_workers=8,
    pipeline_version="0.0.1"
).run()