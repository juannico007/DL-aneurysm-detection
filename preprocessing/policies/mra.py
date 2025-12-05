import itk

from ..policy import ModalityPolicy
from ..normalize import normalize_image
from ..crop import mra_head_crop
from ..registry import register
from ..posthooks import equalize_cta

@register("MRA")
class MraPolicy(ModalityPolicy):
    """Modality policy for MRA volumes."""
    name = "MRA"

    def normalize(self, image: itk.Image, ctx: dict) -> itk.Image:
        """Apply min-max scaling suited for HU intensity ranges."""
        print("working with MRA")
        return normalize_image(image, method="zscore")

    def crop(self, image: itk.Image, ctx: dict) -> itk.Image:
        """Crop the CTA volume to head-only coverage."""
        return mra_head_crop(image)
    
    def post_hooks(self, image: itk.Image, ctx: dict) -> itk.Image:
        """Apply histogram equalization to the region over 1"""
        return equalize_cta(image)
