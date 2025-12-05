from pathlib import Path
import itk
from typing import Tuple

def resample_to_spacing(input_root: Path, series_id: str, voxel_size: Tuple[float, float, float], interpolator = "linear") -> itk.Image:
    """Read ``series_id`` from disk and resample it onto ``voxel_size`` spacing.

    Parameters
    ----------
    input_root : Path
        Root directory containing the raw data; expects a ``series`` subfolder.
    series_id : str
        Folder or file stem that identifies the series to be read.
    voxel_size : tuple[float, float, float]
        Desired output spacing in millimetres (x, y, z order).
    interpolator:
        Linear or spline interpolation
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
    if interpolator == "linear":
        resample.SetInterpolator(itk.LinearInterpolateImageFunction.New(InputImage=image))
    elif interpolator == "spline":
        resample.SetInterpolator(itk.BSplineInterpolateImageFunction.New(input_imaInputImage=image, order=3))
    resample.SetOutputSpacing(voxel_size)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(orig_org)
    resample.SetOutputDirection(orig_dir)
    resample.SetTransform(itk.IdentityTransform[itk.D, 3].New())
    resample.Update()
    return resample.GetOutput()

def resample_to_size(input_root: Path, series_id: str, volume_size: Tuple[float, float, float], interpolator = "linear") -> itk.Image:
    """Read ``series_id`` from disk and resample it onto ``volume``.

    Parameters
    ----------
    input_root : Path
        Root directory containing the raw data; expects a ``series`` subfolder.
    series_id : str
        Folder or file stem that identifies the series to be read.
    vlume_size : tuple[float, float, float]
        Desired output size (x,y,z)
    interpolator:
        Linear or spline interpolation
    Returns
    -------
    itk.Image
        Resampled image with origin and direction preserved from the input.
    """
    print(input_root / "series" / series_id)
    image = itk.imread(input_root / "series" / series_id, itk.F)
    print("image loaded", input_root, series_id)
    input_size = itk.size(image)            # e.g. [x, y, z]
    input_spacing = image.GetSpacing()      # e.g. (sx, sy, sz) 
    orig_org = image.GetOrigin()
    orig_dir = image.GetDirection()
    new_size = volume_size

    itk.OutputWindow.SetGlobalWarningDisplay(False)
    resample = itk.ResampleImageFilter.New(Input=image)
    if interpolator == "linear":
        resample.SetInterpolator(itk.LinearInterpolateImageFunction.New(InputImage=image))
    elif interpolator == "spline":
        resample.SetInterpolator(itk.BSplineInterpolateImageFunction.New(input_imaInputImage=image, order=3))
    
    output_spacing = [
    input_spacing[i] * input_size[i] / new_size[i]
    for i in range(3)
    ]
    resample.SetOutputSpacing(output_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(orig_org)
    resample.SetOutputDirection(orig_dir)
    resample.SetTransform(itk.IdentityTransform[itk.D, 3].New())
    resample.Update()
    return resample.GetOutput()