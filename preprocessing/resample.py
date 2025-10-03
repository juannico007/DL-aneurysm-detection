from pathlib import Path
import itk
from typing import Tuple

def resample_to_spacing(input_root: Path, series_id: str, voxel_size: Tuple[float, float, float]) -> itk.Image:
    """Read ``series_id`` from disk and resample it onto ``voxel_size`` spacing.

    Parameters
    ----------
    input_root : Path
        Root directory containing the raw data; expects a ``series`` subfolder.
    series_id : str
        Folder or file stem that identifies the series to be read.
    voxel_size : tuple[float, float, float]
        Desired output spacing in millimetres (x, y, z order).

    Returns
    -------
    itk.Image
        Resampled image with origin and direction preserved from the input.
    """
    image = itk.imread(input_root / "series" / series_id, itk.F)

    orig_sp = image.GetSpacing()
    orig_sz = image.GetLargestPossibleRegion().GetSize()
    orig_org = image.GetOrigin()
    orig_dir = image.GetDirection()

    new_size = [int(round(osz * osp / nsp)) for osz, osp, nsp in zip(orig_sz, orig_sp, voxel_size)]

    itk.OutputWindow.SetGlobalWarningDisplay(False)
    resample = itk.ResampleImageFilter.New(Input=image)
    resample.SetInterpolator(itk.LinearInterpolateImageFunction.New(InputImage=image))
    resample.SetOutputSpacing(voxel_size)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(orig_org)
    resample.SetOutputDirection(orig_dir)
    resample.SetTransform(itk.IdentityTransform[itk.D, 3].New())
    resample.Update()
    return resample.GetOutput()
