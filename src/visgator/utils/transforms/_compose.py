##
##
##

from dataclasses import dataclass
from typing import Optional

from jaxtyping import UInt8
from torch import Tensor

from visgator.utils.bbox import BBox

from ._transform import Transform


@dataclass(frozen=True, slots=True)
class Compose(Transform):
    """Compose multiple transforms."""

    transforms: list[Transform]
    p: float = 1.0

    def _apply(
        self, img: UInt8[Tensor, "C H W"], bbox: Optional[BBox] = None
    ) -> tuple[UInt8[Tensor, "C H W"], Optional[BBox]]:
        if bbox is None:
            for transform in self.transforms:
                img = transform(img)
            return img, bbox

        if img.shape[1:] != bbox.image_size:
            raise ValueError(
                f"Image size {img.shape[1:]} does not match "
                f"bounding box image size {bbox.image_size}."
            )

        for transform in self.transforms:
            img, bbox = transform(img, bbox)

        return img, bbox
