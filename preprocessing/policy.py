# preprocess/policy.py
import itk


class ModalityPolicy:
    """Base strategy object providing hooks for modality-specific processing.

    Subclasses should override one or more hooks to tailor behaviour for a
    given imaging modality.
    """
    name: str = "BASE"

    def pre_hooks(self, image: itk.Image, ctx: dict) -> itk.Image:
        """Optional early hook executed before cropping and normalisation."""
        return image

    def normalize(self, image: itk.Image, ctx: dict) -> itk.Image:
        """Return an intensity-normalised copy of ``image`` (no-op by default)."""
        return image

    def crop(self, image: itk.Image, ctx: dict) -> itk.Image:
        """Spatially crop ``image`` (no-op by default)."""
        return image
    
    def post_hooks(self, image: itk.Image, ctx: dict) -> itk.Image:
        """Optional late hook executed after cropping and normalisation."""
        return image
