# preprocess/crop.py
import numpy as np
import itk
from itk import array_from_matrix

def ct_head_crop(
    image: itk.Image,
    hu_air_threshold: float = -300.0,
    neck_band: tuple[float, float] = (0.2, 0.8),
    margin: int = 10,
) -> itk.Image:
    """Crop a CT volume to isolate the head region while preserving metadata.

    Parameters
    ----------
    image : itk.Image
        Input CT-like volume (HU intensities) arranged as z, y, x.
    hu_air_threshold : float, default=-300.0
        Minimum HU value treated as tissue.
    neck_band : tuple[float, float], default=(0.2, 0.8)
        Fractional segment of the occupied z-span searched for the neck.
    margin : int, default=10
        Additional slices retained below the detected neck slice.

    Returns
    -------
    itk.Image
        Cropped image with origin, spacing, and direction adjusted accordingly.
    """
    arr = itk.GetArrayFromImage(image).astype(np.float32)  # z, y, x
    tissue = arr > hu_air_threshold
    if tissue.sum() < 1000:
        return image

    area = tissue.reshape(tissue.shape[0], -1).sum(axis=1)
    nz = np.where(area > 0)[0]
    if len(nz) < 10:
        return image

    z_lo, z_hi = nz[0], nz[-1]
    band_lo = z_lo + int((z_hi - z_lo) * neck_band[0])
    band_hi = z_lo + int((z_hi - z_lo) * neck_band[1])
    if band_hi <= band_lo:
        band_lo, band_hi = z_lo, z_hi

    band = area[band_lo:band_hi]
    if band.size == 0:
        return image

    neck_idx = band_lo + int(np.argmin(band))
    z_start = max(neck_idx - margin, 0)
    z_end = z_hi + 1

    submask = tissue[z_start:z_end]
    coords = np.array(np.nonzero(submask))
    zmin, ymin, xmin = coords.min(axis=1)
    zmax, ymax, xmax = coords.max(axis=1) + 1

    cropped = arr[z_start + zmin : z_start + zmax, ymin:ymax, xmin:xmax]
    out = itk.GetImageFromArray(cropped)

    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()

    offset_index = np.array([xmin, ymin, z_start + zmin], dtype=float)  # x,y,z
    spacing_vec = np.asarray(spacing, dtype=float)
    dir_mat = array_from_matrix(direction)
    physical_shift = dir_mat @ (offset_index * spacing_vec)
    new_origin = tuple(np.array(origin) + physical_shift)

    out.SetSpacing(spacing)
    out.SetDirection(direction)
    out.SetOrigin(new_origin)
    return out
