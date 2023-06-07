##
##
##

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from jaxtyping import UInt8
from torch import Tensor
from torchvision.datapoints import BoundingBox

from visgator.utils.bbox import BBox

from ._transform import Transform


@dataclass(frozen=True, slots=True)
class Resize(Transform):
    """Resize an image to a target size.

    This is a wrapper around `torchvision.transforms.functional.resize` that supports
    also bounding boxes. By default, it applies antialiasing to the resized image.

    Attributes
    ----------
    size : Union[int, tuple[int, int]]
        The target size. If an integer is given, the smaller edge of the image
        will be resized to this value and the other edge will be resized keeping
        the aspect ratio. If a tuple is given, the image will be resized to this
        size.
    interpolation : `torchvision.transforms.InterpolationMode`
        The interpolation mode to use when resizing the image.
        Only `InterpolationMode.NEAREST`, `InterpolationMode.NEAREST_EXACT`,
        `InterpolationMode.BILINEAR`, `InterpolationMode.BICUBIC` are supported.
        Defaults to `InterpolationMode.BILINEAR`.
    max_size : Optional[int]
        The maximum size of the larger edge of the resized image. If the larger edge of
        image is larger than this value after being resized according to `size`, then
        `size` will be scaled down so that the larger edge of the resized image is
        equal to this value. As a result, the smaller edge of the resized image will
        be smaller than or equal to `size`.
    antialias : Optional[bool]
        Whether to apply antialiasing to the resized image. Defaults to `True`.
    p : float
        The probability of applying this transform. Defaults to `1.0`.
    """

    size: Union[int, tuple[int, int]]
    interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR
    max_size: Optional[int] = None
    antialias: Optional[bool] = True
    p: float = 1.0

    def _apply(
        self,
        img: UInt8[Tensor, "C H W"],
        bbox: Optional[BBox] = None,
    ) -> tuple[UInt8[Tensor, "C H W"], Optional[BBox]]:
        rescaled: Tensor = F.resize(
            img,
            size=self.size,
            interpolation=self.interpolation,
            max_size=self.max_size,
            antialias=self.antialias,
        )

        if bbox is not None:
            ratio_height = rescaled.shape[1] / img.shape[1]
            ratio_width = rescaled.shape[2] / img.shape[2]
            tensor = bbox.tensor * torch.tensor(
                [ratio_width, ratio_height] * 2,
                device=bbox.tensor.device,
            )

            bbox = BBox(tensor, rescaled.shape[1:], bbox.format, bbox.normalized)

        return rescaled, bbox


@dataclass(init=False, frozen=True, slots=True)
class Rotate(Transform):
    degrees: tuple[float, float]
    interpolation: T.InterpolationMode
    expand: bool
    center: Optional[tuple[float, float]]
    fill: tuple[float, float, float]
    p: float

    def __init__(
        self,
        degrees: Union[float, tuple[float, float]],
        interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
        expand: bool = False,
        center: Optional[tuple[float, float]] = None,
        fill: Union[int, tuple[int, int, int]] = 0,
        p: float = 1.0,
    ):
        if isinstance(degrees, float):
            degrees = (-degrees, degrees)
        if isinstance(fill, int):
            fill = (fill, fill, fill)

        object.__setattr__(self, "degrees", degrees)
        object.__setattr__(self, "interpolation", interpolation)
        object.__setattr__(self, "expand", expand)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "fill", fill)
        object.__setattr__(self, "p", p)

    def _apply(
        self,
        img: UInt8[Tensor, "C H W"],
        bbox: Optional[BBox] = None,
    ) -> tuple[UInt8[Tensor, "C H W"], Optional[BBox]]:
        angle = torch.empty(1).uniform_(*self.degrees).item()

        if bbox is None:
            rotated: Tensor = F.rotate(
                img,
                angle=angle,
                interpolation=self.interpolation,
                expand=self.expand,
                center=self.center,
                fill=self.fill,
            )
        else:
            box = BoundingBox(
                bbox.tensor,
                format=str(bbox.format),
                spatial_size=bbox.image_size,
            )

            img, box = F.rotate(
                img,
                box,
                angle=angle,
                interpolation=self.interpolation,
                expand=self.expand,
                center=self.center,
                fill=self.fill,
            )

            bbox = BBox(box.data, box.spatial_size, bbox.format, bbox.normalized)

        return rotated, bbox


@dataclass(init=False, frozen=True, slots=True)
class Perspective(Transform):
    p: float

    def _apply(
        self,
        img: UInt8[Tensor, "C H W"],
        bbox: Optional[BBox] = None,
    ) -> tuple[UInt8[Tensor, "C H W"], Optional[BBox]]:
        raise NotImplementedError
