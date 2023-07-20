##
##
##

from dataclasses import dataclass
from typing import Literal

import albumentations as A
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torch import Tensor

from deepsight.data.structs import BoundingBoxes, Image, TensorImage

from . import _utils as utils
from ._transformation import Transformation


@dataclass(frozen=True, slots=True)
class Resize(Transformation):
    """Resizes an image to a target size.

    This is a wrapper around `torchvision.transforms.functional.resize` that supports
    also bounding boxes. By default, it applies antialiasing to the resized image.

    Attributes
    ----------
    size : int | tuple[int, int]
        The target size. If an integer is given, the smaller edge of the image
        will be resized to this value and the other edge will be resized keeping
        the aspect ratio. If a tuple is given, the image will be resized to this
        size.
    interpolation : `torchvision.transforms.InterpolationMode`
        The interpolation mode to use when resizing the image.
        Only `InterpolationMode.NEAREST`, `InterpolationMode.NEAREST_EXACT`,
        `InterpolationMode.BILINEAR`, `InterpolationMode.BICUBIC` are supported.
        Defaults to `InterpolationMode.BILINEAR`.
    max_size : int | None
        The maximum size of the larger edge of the resized image. If the larger edge of
        image is larger than this value after being resized according to `size`, then
        `size` will be scaled down so that the larger edge of the resized image is
        equal to this value. As a result, the smaller edge of the resized image will
        be smaller than or equal to `size`. Defaults to `None`.
    antialias : bool | None
        Whether to apply antialiasing to the resized image. Defaults to `True`.
    p : float
        The probability of applying this transform. Defaults to `1.0`.
    """

    size: int | tuple[int, int]
    interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR
    max_size: int | None = None
    antialias: bool | None = None
    p: float = 1.0

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        if torch.rand(1) > self.p:
            return image, boxes

        if boxes is not None:
            utils.check_boxes(boxes, image.size)

        image = image.to_tensor()
        rescaled: Tensor = F.resize(
            image.data,
            size=self.size,
            interpolation=self.interpolation,
            max_size=self.max_size,
            antialias=self.antialias,
        )
        rescaled_image = TensorImage(rescaled, image.normalized)

        if boxes is not None:
            boxes = boxes.normalize()
            rescaled_boxes = BoundingBoxes(
                boxes.tensor,
                rescaled_image.size,
                format=boxes.format,
                normalized=True,
            )

            return rescaled_image, rescaled_boxes

        return rescaled_image, None


class Perspective(Transformation):
    def __init__(
        self,
        scale: float | tuple[float, float] = (0.05, 0.1),
        keep_size: bool = True,
        pad_mode: int = 0,
        pad_value: int | float | list[int] | list[float] = 0,
        mask_pad_val: int | float | list[int] | list[float] = 0,
        fit_output: bool = False,
        interpolation: int = 1,
        p: float = 0.5,
    ) -> None:
        super().__init__()

        transform = A.Perspective(
            scale=scale,
            keep_size=keep_size,
            pad_mode=pad_mode,
            pad_value=pad_value,
            mask_pad_val=mask_pad_val,
            fit_output=fit_output,
            interpolation=interpolation,
            p=p,
        )

        self._transform = A.Compose(
            [transform], bbox_params=A.BboxParams(format="pascal_voc"), p=1.0
        )

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        return utils.apply_albumentation_transform(self._transform, image, boxes)


class ElasticTransform(Transformation):
    def __init__(
        self,
        alpha: float = 1,
        sigma: float = 50,
        alpha_affine: float = 50,
        interpolation: int = 1,
        border_mode: int = 4,
        value: int | float | list[int] | list[float] | None = None,
        approximate: bool = False,
        same_dxdy: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__()

        transform = A.ElasticTransform(
            alpha=alpha,
            sigma=sigma,
            alpha_affine=alpha_affine,
            interpolation=interpolation,
            border_mode=border_mode,
            value=value,
            approximate=approximate,
            same_dxdy=same_dxdy,
            p=p,
        )

        self._transform = A.Compose(
            [transform], bbox_params=A.BboxParams(format="pascal_voc"), p=1.0
        )

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        return utils.apply_albumentation_transform(self._transform, image, boxes)


class Rotate(Transformation):
    def __init__(
        self,
        limit: int | tuple[int, int] = 90,
        interpolation: int = 1,
        border_mode: int = 4,
        value: int | float | list[int] | list[float] | None = None,
        rotate_mode: Literal["largest_box", "ellipse"] = "largest_box",
        crop_border: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__()

        transform = A.Rotate(
            limit=limit,
            interpolation=interpolation,
            border_mode=border_mode,
            value=value,
            rotate_mode=rotate_mode,
            crop_border=crop_border,
            p=p,
        )

        self._transform = A.Compose(
            [transform], bbox_params=A.BboxParams(format="pascal_voc"), p=1.0
        )

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        return utils.apply_albumentation_transform(self._transform, image, boxes)


class PiecewiseAffine(Transformation):
    def __init__(
        self,
        scale: float | tuple[float, float] = (0.03, 0.05),
        nb_rows: int | tuple[int, int] = 4,
        nb_cols: int | tuple[int, int] = 4,
        interpolation: int = 1,
        cval: int | float = 0,
        mode: Literal["constant", "edge", "symmetric", "reflect", "wrap"] = "constant",
        absolute_scale: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__()

        transform = A.PiecewiseAffine(
            scale=scale,
            nb_rows=nb_rows,
            nb_cols=nb_cols,
            interpolation=interpolation,
            cval=cval,
            mode=mode,
            absolute_scale=absolute_scale,
            p=p,
        )

        self._transform = A.Compose(
            [transform], bbox_params=A.BboxParams(format="pascal_voc"), p=1.0
        )

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        return utils.apply_albumentation_transform(self._transform, image, boxes)


class Affine(Transformation):
    def __init__(
        self,
        scale: int | float | tuple[int | float, int | float] | None = None,
        translate_percent: int | float | tuple[int | float, int | float] | None = None,
        translate_px: int | float | tuple[int | float, int | float] | None = None,
        rotate: int | float | tuple[int | float, int | float] | None = None,
        shear: int | float | tuple[int | float, int | float] | None = None,
        interpolation: int = 1,
        cval: int | float | list[int] | list[float] = 0,
        mode: int = 0,
        fit_output: bool = False,
        keep_ratio: bool = False,
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box",
        p: float = 0.5,
    ) -> None:
        super().__init__()

        transform = A.Affine(
            scale=scale,
            translate_percent=translate_percent,
            translate_px=translate_px,
            rotate=rotate,
            shear=shear,
            interpolation=interpolation,
            cval=cval,
            mode=mode,
            fit_output=fit_output,
            keep_ratio=keep_ratio,
            rotate_method=rotate_method,
            p=p,
        )

        self._transform = A.Compose(
            [transform], bbox_params=A.BboxParams(format="pascal_voc"), p=1.0
        )

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        return utils.apply_albumentation_transform(self._transform, image, boxes)
