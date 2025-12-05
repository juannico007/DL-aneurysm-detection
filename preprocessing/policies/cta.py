import itk

from ..policy import ModalityPolicy
from ..normalize import normalize_image
from ..crop import ct_head_crop
from ..registry import register
from ..posthooks import equalize_cta

@register("CTA")
class CtaPolicy(ModalityPolicy):
    """Modality policy for CT angiography (CTA) volumes."""
    name = "CTA"

    def normalize(self, image: itk.Image, ctx: dict) -> itk.Image:
        """Apply min-max scaling suited for HU intensity ranges."""
        return normalize_image(image, method="minmax")

    def crop(self, image: itk.Image, ctx: dict) -> itk.Image:
        """Crop the CTA volume to head-only coverage."""
        return ct_head_crop(image)
    
    def post_hooks(self, image: itk.Image, ctx: dict) -> itk.Image:
        """Apply histogram equalization to the region over 1"""
        return equalize_cta(image)
