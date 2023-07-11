##
##
##

from dataclasses import dataclass

import albumentations as A
import torch
import torchvision.transforms.v2.functional as F

from deepsight.data.structs import BoundingBoxes, Image, NumpyImage, TensorImage

from . import _utils as utils
from ._transformation import Transformation


@dataclass(frozen=True, slots=True)
class Standardize(Transformation):
    """Standardizes the input image.

    Attributes
    ----------
    mean : tuple[float, float, float]
        The mean of the input image.
    std : tuple[float, float, float]
        The standard deviation of the input image.
    p : float, optional
        The probability of applying the transformation. Defaults to 1.0.
    """

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    p: float = 1.0

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        if torch.rand(1) > self.p:
            return image, boxes

        image = image.to_tensor().normalize()
        standardized = F.normalize_image_tensor(image.data, self.mean, self.std)
        return TensorImage(standardized, True), boxes


class ColorJitter(Transformation):
    def __init__(
        self,
        brightness: float | tuple[float, float] = 0.2,
        contrast: float | tuple[float, float] = 0.2,
        saturation: float | tuple[float, float] = 0.2,
        hue: float | tuple[float, float] = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__()

        self._transform = A.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=p,
        )

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        return utils.apply_albumentation_transform(self._transform, image, boxes)


class Posterize(Transformation):
    def __init__(
        self,
        num_bits: int | tuple[int, int] | list[int] | list[tuple[int, int]] = 4,
        p: float = 0.5,
    ) -> None:
        super().__init__()

        self._transform = A.Posterize(num_bits=num_bits, p=p)

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        image = image.to_numpy()
        transformed = self._transform(image=image)["image"]
        return NumpyImage(transformed), boxes


class GaussianNoise(Transformation):
    def __init__(
        self,
        var_limit: float | tuple[float, float] = (10.0, 50.0),
        mean: float = 0.0,
        per_channel: bool = True,
        p: float = 0.5,
    ) -> None:
        super().__init__()

        self._transform = A.GaussianNoise(
            var_limit=var_limit,
            mean=mean,
            per_channel=per_channel,
            p=p,
        )

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        return utils.apply_albumentation_transform(self._transform, image, boxes)


class GlassBlur(Transformation):
    def __init__(
        self,
        sigma: float = 0.7,
        max_delta: int = 4,
        iterations: int = 2,
        mode: str = "fast",
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self._transform = A.GlassBlur(
            sigma=sigma,
            max_delta=max_delta,
            iterations=iterations,
            mode=mode,
            p=p,
        )

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        return utils.apply_albumentation_transform(self._transform, image, boxes)


class RandomToneCurve(Transformation):
    def __init__(self, scale: float = 0.1, p: float = 0.5) -> None:
        super().__init__()

        self._transform = A.RandomToneCurve(scale=scale, p=p)

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        return utils.apply_albumentation_transform(self._transform, image, boxes)


class RandomBrightnessContrast(Transformation):
    def __init__(
        self,
        brightness_limit: float | tuple[float, float] = 0.2,
        contrast_limit: float | tuple[float, float] = 0.2,
        brightness_by_max: bool = True,
        p: float = 0.5,
    ) -> None:
        super().__init__()

        self._transform = A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            brightness_by_max=brightness_by_max,
            p=p,
        )

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        return utils.apply_albumentation_transform(self._transform, image, boxes)


class Defocus(Transformation):
    def __init__(
        self,
        radius: int | tuple[int, int] = (3, 10),
        alias_blur: float | tuple[float, float] = (0.1, 0.5),
    ) -> None:
        super().__init__()

        self._transform = A.Defocus(radius=radius, alias_blur=alias_blur)

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        return utils.apply_albumentation_transform(self._transform, image, boxes)


class RandomGamma(Transformation):
    def __init__(
        self,
        gamma_limit: float | tuple[float, float] = (80.0, 120.0),
        p: float = 0.5,
    ) -> None:
        super().__init__()

        self._transform = A.RandomGamma(gamma_limit=gamma_limit, p=p)

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        return utils.apply_albumentation_transform(self._transform, image, boxes)
