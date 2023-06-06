##
##
##

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torchvision.transforms.functional as F
from jaxtyping import UInt8
from torch import Tensor

from visgator.utils.bbox import BBox

from ._transform import Transform


@dataclass(init=False, frozen=True, slots=True)
class GaussianNoise(Transform):
    """Add Gaussian noise to an image."""

    mean: float
    std: tuple[float, float]
    per_channel: bool
    p: float

    def __init__(
        self,
        mean: float = 0.0,
        std: Union[float, tuple[float, float]] = (3, 7),
        per_channel: bool = True,
        p: float = 0.5,
    ):
        if isinstance(std, float):
            std = (std, std)

        if std[0] < 0 or std[1] < 0:
            raise ValueError("Standard deviation must be non-negative.")

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "per_channel", per_channel)
        object.__setattr__(self, "p", p)

    def _apply(
        self, img: Tensor, bbox: Optional[BBox] = None
    ) -> tuple[UInt8[Tensor, "C H W"], Optional[BBox]]:
        std = torch.empty(1).uniform_(self.std[0], self.std[1]).item()

        if self.per_channel:
            noise = torch.normal(self.mean, std, size=img.shape)
        else:
            noise = torch.normal(self.mean, std, size=img.shape[1:])

        img = img + noise
        img = torch.clamp(img, 0, 255).to(torch.uint8)

        return img, bbox


@dataclass(init=False, frozen=True, slots=True)
class GaussianBlur(Transform):
    """Apply Gaussian blur to an image."""

    size: tuple[int, int]
    std: tuple[float, float]
    p: float

    def __init__(
        self,
        size: Union[int, tuple[int, int]] = (3, 7),
        std: Union[float, tuple[float, float]] = 0.0,
        p: float = 0.5,
    ):
        size = (size, size) if isinstance(size, int) else size
        std = (std, std) if isinstance(std, float) else std

        if size == (0, 0) and std == (0, 0):
            raise ValueError("Either size or std must be non-zero.")

        match size, std:
            case (0, 0), (0, 0):
                raise ValueError("Either size or std must be non-zero.")
            case (0, 0), (std_0, std_1):
                size_a = round(std_0 * 6 + 1) + 1
                size_b = round(std_1 * 6 + 1) + 1
                size = (size_a, size_b)
            case (size_0, size_1), (0, 0):
                std_a = ((size_0 - 1) * 0.5 - 1) + 0.8
                std_b = ((size_1 - 1) * 0.5 - 1) + 0.8
                std = (std_a, std_b)

        if size[0] % 2 == 0 or size[1] % 2 == 0:
            raise ValueError("Size must be odd.")

        if std[0] < 0 or std[1] < 0:
            raise ValueError("Standard deviation must be non-negative.")

        object.__setattr__(self, "size", size)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "p", p)

    def _apply(
        self, img: Tensor, bbox: Optional[BBox] = None
    ) -> tuple[UInt8[Tensor, "C H W"], Optional[BBox]]:
        size = torch.empty(2).uniform_(self.size[0], self.size[1] + 1).int().tolist()
        std = torch.empty(2).uniform_(self.std[0], self.std[1]).tolist()

        img = F.gaussian_blur(img, size, std)

        return img, bbox


@dataclass(init=False, frozen=True, slots=True)
class Posterize(Transform):
    bits: tuple[int, int]
    p: float

    def __init__(self, bits: Union[int, tuple[int, int]] = (4, 8), p: float = 0.5):
        bits = (bits, bits) if isinstance(bits, int) else bits

        if not (0 <= bits[0] <= 8 and 0 <= bits[1] <= 8):
            raise ValueError("Bits must be between 0 and 8.")

        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "p", p)

    def _apply(
        self, img: Tensor, bbox: Optional[BBox] = None
    ) -> tuple[UInt8[Tensor, "C H W"], Optional[BBox]]:
        bits = torch.empty(2).uniform_(self.bits[0], self.bits[1] + 1).int().tolist()
        img = F.posterize(img, bits)

        return img, bbox


@dataclass(frozen=True, slots=True)
class Equalize(Transform):
    p: float

    def _apply(
        self, img: Tensor, bbox: Optional[BBox] = None
    ) -> tuple[UInt8[Tensor, "C H W"], Optional[BBox]]:
        img = F.equalize(img)

        return img, bbox
