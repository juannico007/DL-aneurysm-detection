# Put this in your policy.post_hooks or import it there
import itk
import numpy as np
import gc
import gc
from typing import Optional, Tuple
from skimage import exposure

def equalize_cta(itk_image: itk.Image,
                    lower_pct: float = 10.0,
                    upper_pct: float = 90.0,
                    clip_limit: float = 0.01,
                    kernel_size: int | tuple[int, int, int] | None = None,
                    mask_threshold: Optional[float] = 0.01) -> itk.Image:
    """
    Apply CLAHE using scikit-image on a NumPy volume extracted from an ITK image.
    Assumes image is already normalized in [0,1].

    Args:
        itk_image: ITK image (3D)
        clip_limit: scikit-image's clip_limit (typical 0.01 - 0.03 for medical)
        kernel_size: neighborhood size; int or tuple. If None, autocomputed.

    Returns:
        itk.Image: equalized image with original metadata preserved.
    """
    arr = itk.GetArrayFromImage(itk_image).astype(np.float32)
    # Build foreground mask to avoid using the background for the equalization
    if mask_threshold is not None:
        fg_mask = arr > float(mask_threshold)  #Treats everything < mask_threshold as background for the equalization
    else:
        # non-zero mask; good if cropping removes background, but if zeros exist inside head, consider mask_threshold
        fg_mask = arr != 0

    # Compute percentiles on foreground only
    fg_vals = arr[fg_mask]
    lo = np.percentile(fg_vals, lower_pct)
    hi = np.percentile(fg_vals, upper_pct)

    # Clip, its better to keep the intensities between those percentiles in our volumes since its distribution are too skewed
    clipped = np.clip(arr, lo, hi)

    # Decide kernel size (in voxels). If None use ~1/8 of image size per axis
    if kernel_size is None:
        ks = tuple(max(1, s // 8) for s in arr.shape)  # (z, y, x)
    elif isinstance(kernel_size, int):
        ks = (kernel_size, kernel_size, kernel_size)
    else:
        ks = kernel_size
    # scikit-image expects array in (z,y,x) or (y,x) etc. Works for 3D when given a 3D array.
    # apply equalize_adapthist (returns floats in [0,1])
    equalized = exposure.equalize_adapthist(clipped, kernel_size=ks, clip_limit=clip_limit)
    
    # back to ITK image
    out_itk = itk.GetImageFromArray(equalized.astype(np.float32))
    out_itk.CopyInformation(itk_image)
    
    hi = np.percentile(fg_vals, upper_pct)

    return out_itk