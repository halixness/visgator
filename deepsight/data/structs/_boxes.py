##
##
##

import enum
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor
from typing_extensions import Self

from deepsight.utils.protocols import Moveable


class BoundingBoxFormat(enum.Enum):
    """Coordinates format of a bounding box."""

    XYXY = "xyxy"
    XYWH = "xywh"
    CXCYWH = "cxcywh"


@dataclass(init=False, frozen=True, slots=True)
class BoundingBoxes(Moveable):
    """A collection of bounding boxes with their corresponding images size.

    Attributes
    ----------
    tensor : Float[Tensor, "... 4"]
        The boxes tensor.
    images_size : Uint[Tensor, "... 2"]
        The images size tensor.
    format : BoxFormat
        The boxes format.
    normalized : bool
        Whether the boxes are normalized or not.
    """

    # -------------------------------------------------------------------------
    # Attributes and properties
    # -------------------------------------------------------------------------

    tensor: Float[Tensor, "... 4"]
    images_size: Int[Tensor, "... 2"]
    format: BoundingBoxFormat
    normalized: bool

    @property
    def device(self) -> torch.device:
        """The device of the boxes."""
        return self.tensor.device

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(
        self,
        tensor: Float[Tensor, "... 4"],
        images_size: Int[Tensor, "... 2"] | tuple[int, int],
        format: BoundingBoxFormat,
        normalized: bool,
    ):
        """Initializes a new `BoundingBoxes` instance.

        Parameters
        ----------
        tensor : Float[Tensor, "... 4"]
            The boxes tensor.
        images_size : Int[Tensor, "... 2"] | tuple[int, int]
            The images size tensor or a tuple containing the images size.
            If a tuple is given, it will be considered as the images size of
            all the boxes.
        format : BoxFormat
            The boxes format.
        normalized : bool
            Whether the boxes are normalized or not.
        """

        if isinstance(images_size, tuple):
            images_size = torch.tensor(images_size, device=tensor.device)
            shape = tensor.shape[:-1] + (-1,)
            images_size = images_size.expand(shape)

        if tensor.size(-1) != 4:
            raise ValueError(
                "Tensor should have 4 elements in the last dimension, "
                f"got {tensor.size(-1)}."
            )

        if images_size.size(-1) != 2:
            raise ValueError(
                "Images size should have 2 elements in the last dimension, "
                f"got {images_size.size(-1)}."
            )

        if tensor.shape[:-1] != images_size.shape[:-1]:
            raise ValueError(
                "Tensor and images size should have the same shape, "
                f"got {tensor.shape[:-1]} and {images_size.shape[:-1]}."
            )

        if tensor.device != images_size.device:
            raise ValueError(
                "Tensor and images size should be on the same device, "
                f"got {tensor.device} and {images_size.device}."
            )

        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "images_size", images_size)
        object.__setattr__(self, "format", format)
        object.__setattr__(self, "normalized", normalized)

    @classmethod
    def cat(cls, boxes: list[Self], dim: int = 0) -> Self:
        """Concatenates the boxes along the given dimension.

        Parameters
        ----------
        boxes : list[BoundingBoxes]
            The boxes to concatenate. All the boxes should have the same format
            and whether they are normalized or not.
        dim : int, optional
            The dimension to concatenate along. Defaults to 0.

        Returns
        -------
        BoundingBoxes
            The concatenated boxes.
        """

        cls._check_list(boxes)

        tensor = torch.cat([box.tensor for box in boxes], dim=dim)
        images_size = torch.cat([box.images_size for box in boxes], dim=dim)

        return cls(
            tensor=tensor,
            images_size=images_size,
            format=boxes[0].format,
            normalized=boxes[0].normalized,
        )

    @classmethod
    def stack(cls, boxes: list[Self], dim: int = 0) -> Self:
        """Stacks the boxes along the given dimension.

        Parameters
        ----------
        boxes : list[BoundingBoxes]
            The boxes to stack. All the boxes should have the same format and
            whether they are normalized or not.
        dim : int, optional
            The dimension to stack along. Defaults to 0.

        Returns
        -------
        BoundingBoxes
            The stacked boxes.
        """

        cls._check_list(boxes)

        tensor = torch.stack([box.tensor for box in boxes], dim=dim)
        images_size = torch.stack([box.images_size for box in boxes], dim=dim)

        return cls(
            tensor=tensor,
            images_size=images_size,
            format=boxes[0].format,
            normalized=boxes[0].normalized,
        )

    @classmethod
    def pad_sequence(
        cls,
        boxes: list[Self],
        pad_value: float = 0.0,
        pad_image_size: float | None = None,
    ) -> Self:
        """Pads the boxes to the same length along the first dimension
        and stacks them.

        Given a list of boxes of size :math:: `(M_i, ...)`, where `M` can be different
        for each box, this function will pad the boxes and images size to the same
        length along the first dimension and stack them, resulting in a tensor of size
        `(N, M, ...)`, where `N` is the number of boxes and `M` is the maximum `M_i`.

        Parameters
        ----------
        boxes : list[BoundingBoxes]
            The boxes to pad. All the boxes should have the same format and
            whether they are normalized or not. All boxes should also have the
            same number of dimensions and the dimensions should be the same
            except for the first one.
        pad_value : float, optional
            The value to use for padding. Defaults to `0.0`.
        pad_image_size : float | None, optional
            The value to use for padding the images size. If `None` and all the i^th
            images size are the same, the i^th images size will be used for all the
            padded boxes in the i^th position. If `None` and the i^th images size are
            not the same, an error will be raised. Defaults to `None`.

        Returns
        -------
        BoundingBoxes
            The padded boxes.
        """

        cls._check_list(boxes)

        max_length = max(box.tensor.size(0) for box in boxes)
        other_dims = boxes[0].tensor.shape[1:-1]

        tensor = boxes[0].tensor.new_full(
            (len(boxes), max_length, *other_dims, 4),
            fill_value=pad_value,
        )

        images_size = boxes[0].images_size.new_empty(
            (len(boxes), max_length, *other_dims, 2),
        )

        for i, box in enumerate(boxes):
            tensor[i, : box.tensor.size(0)].copy_(box.tensor)

            if pad_image_size is None:
                if not torch.all(box.images_size == box.images_size[0]):
                    raise ValueError(
                        "All images size should be the same if pad_image_size is None."
                    )
                else:
                    images_size[i, ..., :].copy_(box.images_size[0])
            else:
                images_size[i, : box.tensor.size(0)].copy_(box.images_size)
                images_size[i, box.tensor.size(0) :, ...].fill_(pad_image_size)

        return cls(
            tensor=tensor,
            images_size=images_size,
            format=boxes[0].format,
            normalized=boxes[0].normalized,
        )

    @classmethod
    def _check_list(cls, boxes: list[Self]) -> None:
        if len(boxes) == 0:
            raise ValueError("BoundingBoxes list should not be empty.")

        if not all(box.normalized == boxes[0].normalized for box in boxes):
            raise ValueError("All boxes should have the same normalized value.")

        if not all(box.format == boxes[0].format for box in boxes):
            raise ValueError("All boxes should have the same format.")

    # -------------------------------------------------------------------------
    # Operation on self
    # -------------------------------------------------------------------------

    def to_xyxy(self) -> Self:
        """Converts the boxes to the (x1, y1, x2, y2) format.

        Returns
        -------
        BoundingBoxes
            The converted boxes.
        """

        match self.format:
            case BoundingBoxFormat.XYXY:
                return self
            case BoundingBoxFormat.XYWH:
                x1y1 = self.tensor[..., :2]
                x2y2 = self.tensor[..., :2] + self.tensor[..., 2:]
                tensor = torch.cat([x1y1, x2y2], dim=-1)
            case BoundingBoxFormat.CXCYWH:
                x1y1 = self.tensor[..., :2] - self.tensor[..., 2:] / 2
                x2y2 = self.tensor[..., :2] + self.tensor[..., 2:] / 2
                tensor = torch.cat([x1y1, x2y2], dim=-1)

        return self.__class__(
            tensor=tensor,
            images_size=self.images_size,
            format=BoundingBoxFormat.XYXY,
            normalized=self.normalized,
        )

    def to_xywh(self) -> Self:
        """Converts the boxes to the (x, y, w, h) format.

        Returns
        -------
        BoundingBoxes
            The converted boxes.
        """

        match self.format:
            case BoundingBoxFormat.XYXY:
                xy = self.tensor[..., :2]
                wh = self.tensor[..., 2:] - self.tensor[..., :2]
                tensor = torch.cat([xy, wh], dim=-1)
            case BoundingBoxFormat.XYWH:
                return self
            case BoundingBoxFormat.CXCYWH:
                xy = self.tensor[..., :2] - self.tensor[..., 2:] / 2
                wh = self.tensor[..., 2:]
                tensor = torch.cat([xy, wh], dim=-1)

        return self.__class__(
            tensor=tensor,
            images_size=self.images_size,
            format=BoundingBoxFormat.XYWH,
            normalized=self.normalized,
        )

    def to_cxcywh(self) -> Self:
        """Converts the boxes to the (cx, cy, w, h) format.

        Returns
        -------
        BoundingBoxes
            The converted boxes.
        """

        match self.format:
            case BoundingBoxFormat.XYXY:
                cxcy = (self.tensor[..., :2] + self.tensor[..., 2:]) / 2
                wh = self.tensor[..., 2:] - self.tensor[..., :2]
                tensor = torch.cat([cxcy, wh], dim=-1)
            case BoundingBoxFormat.XYWH:
                cxcy = self.tensor[..., :2] + self.tensor[..., 2:] / 2
                wh = self.tensor[..., 2:]
                tensor = torch.cat([cxcy, wh], dim=-1)
            case BoundingBoxFormat.CXCYWH:
                return self

        return self.__class__(
            tensor=tensor,
            images_size=self.images_size,
            format=BoundingBoxFormat.CXCYWH,
            normalized=self.normalized,
        )

    def to_format(self, format: BoundingBoxFormat) -> Self:
        """Converts the boxes to the given format.

        Parameters
        ----------
        format : BoundingBoxFormat
            The format to convert to.

        Returns
        -------
        BoundingBoxes
            The converted boxes.
        """

        match format:
            case BoundingBoxFormat.XYXY:
                return self.to_xyxy()
            case BoundingBoxFormat.XYWH:
                return self.to_xywh()
            case BoundingBoxFormat.CXCYWH:
                return self.to_cxcywh()

    def normalize(self) -> Self:
        """Normalizes the boxes to the [0, 1] range.

        Returns
        -------
        BoundingBoxes
            The normalized boxes.
        """

        if self.normalized:
            return self

        tensor = self.tensor.clone()
        tensor[..., 0::2] /= self.images_size[..., 1].unsqueeze(-1)
        tensor[..., 1::2] /= self.images_size[..., 0].unsqueeze(-1)

        return self.__class__(
            tensor=tensor,
            images_size=self.images_size,
            format=self.format,
            normalized=True,
        )

    def denormalize(self) -> Self:
        """Denormalizes the boxes to the corresponding image range.

        Returns
        -------
        BoundingBoxes
            The denormalized boxes.
        """

        if not self.normalized:
            return self

        tensor = self.tensor.clone()
        tensor[..., 0::2] *= self.images_size[..., 1].unsqueeze(-1)
        tensor[..., 1::2] *= self.images_size[..., 0].unsqueeze(-1)

        return self.__class__(
            tensor=tensor,
            images_size=self.images_size,
            format=self.format,
            normalized=False,
        )

    def area(self) -> Float[Tensor, "..."]:
        """Computes the area of the boxes.

        Returns
        -------
        Float[Tensor, "..."]
            The area of the boxes.
        """

        match self.format:
            case BoundingBoxFormat.XYXY:
                w = self.tensor[..., 2] - self.tensor[..., 0]
                h = self.tensor[..., 3] - self.tensor[..., 1]
                return w * h
            case BoundingBoxFormat.XYWH:
                return self.tensor[..., 2] * self.tensor[..., 3]
            case BoundingBoxFormat.CXCYWH:
                return self.tensor[..., 2] * self.tensor[..., 3]

    # -------------------------------------------------------------------------
    # Operations between self and other
    # -------------------------------------------------------------------------

    def pairwise_check(self, other: Self) -> None:
        """Checks that the boxes have the same shape, images size, normalization
        and format.

        Parameters
        ----------
        other : BoundingBoxes
            The other boxes.

        Raises
        ------
        ValueError
            If the boxes do not have the same shape, images size, normalization
            or format.
        """

        if self.tensor.shape != other.tensor.shape:
            raise ValueError(
                "The boxes should have the same shape, but got "
                f"{self.tensor.shape} and {other.tensor.shape}."
            )

        if torch.any(self.images_size != other.images_size):
            raise ValueError("The boxes should have the same images size.")

        if self.normalized != other.normalized:
            raise ValueError("The boxes should have the same normalization.")

        if self.format != other.format:
            raise ValueError(
                "The boxes should have the same format, but got "
                f"{self.format} and {other.format}."
            )

    def __or__(self, other: Self) -> Self:
        boxes1 = self.to_xyxy()
        boxes2 = other.to_xyxy()
        if self.normalized != other.normalized:
            normalized = True
            boxes1 = boxes1.normalize()
            boxes2 = boxes2.normalize()
        else:
            normalized = self.normalized

        boxes1.pairwise_check(boxes2)

        x1y1 = torch.min(boxes1.tensor[..., :2], boxes2.tensor[..., :2])
        x2y2 = torch.max(boxes1.tensor[..., 2:], boxes2.tensor[..., 2:])

        tensor = torch.cat([x1y1, x2y2], dim=-1)
        return self.__class__(
            tensor=tensor,
            images_size=self.images_size,
            format=BoundingBoxFormat.XYXY,
            normalized=normalized,
        )

    def __and__(self, other: Self) -> Self:
        boxes1 = self.to_xyxy()
        boxes2 = other.to_xyxy()
        if self.normalized != other.normalized:
            normalized = True
            boxes1 = boxes1.normalize()
            boxes2 = boxes2.normalize()
        else:
            normalized = self.normalized

        boxes1.pairwise_check(boxes2)

        x1y1 = torch.max(boxes1.tensor[..., :2], boxes2.tensor[..., :2])
        x2y2 = torch.min(boxes1.tensor[..., 2:], boxes2.tensor[..., 2:])

        wh = torch.clamp(x2y2 - x1y1, min=0)
        tensor = torch.cat([x1y1, wh], dim=-1)

        return self.__class__(
            tensor=tensor,
            images_size=self.images_size,
            format=BoundingBoxFormat.XYWH,
            normalized=normalized,
        )

    def union(self, other: Self) -> Self:
        """Computes the pairwise union of the boxes.

        Parameters
        ----------
        other : BoundingBoxes
            The other boxes. Should have the same shape and images size.

        Returns
        -------
        BoundingBoxes
            The pairwise union of the boxes.
        """

        return self | other

    def intersection(self, other: Self) -> Self:
        """Computes the pairwise intersection of the boxes.

        If the boxes are not overlapping, the intersection is a degerate box
        with zero width and height.

        Parameters
        ----------
        other : BoundingBoxes
            The other boxes. Should have the same shape and images size.

        Returns
        -------
        BoundingBoxes
            The pairwise intersection of the boxes.
        """

        return self & other

    def union_area(self, other: Self) -> Float[Tensor, "..."]:
        """Computes the pairwise union area of the boxes.

        .. note::
            This is not the same as the area of the union of the boxes.

        Parameters
        ----------
        other : BoundingBoxes
            The other boxes. Should have the same shape and images size.

        Returns
        -------
        Float[Tensor, "..."]
            The pairwise union area of the boxes.
        """

        if self.normalized != other.normalized:
            raise ValueError("The boxes should have the same normalization.")

        area1 = self.area()
        area2 = other.area()
        intersection_area = self.intersection_area(other)

        return area1 + area2 - intersection_area

    def intersection_area(self, other: Self) -> Float[Tensor, "..."]:
        """Computes the pairwise intersection area of the boxes.

        Parameters
        ----------
        other : BoundingBoxes
            The other boxes. Should have the same shape and images size.

        Returns
        -------
        Float[Tensor, "..."]
            The pairwise intersection area of the boxes.
        """

        intersection = self.intersection(other)
        return intersection.area()

    def iou(self, other: Self) -> Float[Tensor, "..."]:
        """Computes the pairwise IoU of the boxes.

        Parameters
        ----------
        other : BoundingBoxes
            The other boxes. Should have the same shape and images size.

        Returns
        -------
        Float[Tensor, "..."]
            The pairwise IoU of the boxes.
        """

        boxes1 = self.normalize()
        boxes2 = other.normalize()

        intersection_area = boxes1.intersection_area(boxes2)
        union_area = boxes1.union_area(boxes2)

        eps = torch.finfo(intersection_area.dtype).eps
        return intersection_area / (union_area + eps)

    def generalized_iou(self, other: Self) -> Float[Tensor, "..."]:
        """Computes the pairwise generalized IoU of the boxes.

        Parameters
        ----------
        other : BoundingBoxes
            The other boxes. Should have the same shape and images size.

        Returns
        -------
        Float[Tensor, "..."]
            The pairwise generalized IoU of the boxes.
        """

        boxes1 = self.normalize()
        boxes2 = other.normalize()

        intersection_area = boxes1.intersection_area(boxes2)
        union_area = boxes1.union_area(boxes2)
        total_area = boxes1.union(boxes2).normalize().area()

        eps = torch.finfo(self.tensor.dtype).eps

        iou = intersection_area / (union_area + eps)
        giou = iou - (total_area - union_area) / (total_area + eps)

        return giou

    # -------------------------------------------------------------------------
    # PyTorch methods
    # -------------------------------------------------------------------------

    def to(self, device: torch.device | str) -> Self:
        """Moves the boxes to the specified device.

        Parameters
        ----------
        device : torch.device | str
            The device to move the boxes to.

        Returns
        -------
        BoundingBoxes
            The moved boxes.
        """

        return self.__class__(
            tensor=self.tensor.to(device),
            images_size=self.images_size.to(device),
            format=self.format,
            normalized=self.normalized,
        )

    def numel(self) -> int:
        """Returns the number of boxes.

        Returns
        -------
        int
            The number of boxes.
        """

        return int(self.tensor.numel()) // 4

    def unsqueeze(self, dim: int) -> Self:
        """Unsqueezes the boxes along the specified dimension.

        Parameters
        ----------
        dim : int
            The dimension to unsqueeze.

        Returns
        -------
        BoundingBoxes
            The unsqueezed boxes.
        """

        return self.__class__(
            tensor=self.tensor.unsqueeze(dim=dim),
            images_size=self.images_size.unsqueeze(dim=dim),
            format=self.format,
            normalized=self.normalized,
        )

    def squeeze(self, dim: int) -> Self:
        """Squeezes the boxes along the specified dimension.

        Parameters
        ----------
        dim : int
            The dimension to squeeze.

        Returns
        -------
        BoundingBoxes
            The squeezed boxes.
        """

        return self.__class__(
            tensor=self.tensor.squeeze(dim=dim),
            images_size=self.images_size.squeeze(dim=dim),
            format=self.format,
            normalized=self.normalized,
        )

    def expand(self, *sizes: int) -> Self:
        """Expands the boxes to the specified size.

        Parameters
        ----------
        *sizes : int
            The size to expand to.

        Returns
        -------
        BoundingBoxes
            The expanded boxes.
        """

        return self.__class__(
            tensor=self.tensor.expand(*sizes),
            images_size=self.images_size.expand(*sizes),
            format=self.format,
            normalized=self.normalized,
        )

    def repeat(self, *sizes: int) -> Self:
        """Repeats the boxes to the specified size.

        Parameters
        ----------
        *sizes : int
            The size to repeat to.

        Returns
        -------
        BoundingBoxes
            The repeated boxes.
        """

        return self.__class__(
            tensor=self.tensor.repeat(*sizes),
            images_size=self.images_size.repeat(*sizes),
            format=self.format,
            normalized=self.normalized,
        )

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Self:
        """Flattens the boxes.

        Parameters
        ----------
        start_dim : int, optional
            The start dimension to flatten. Defaults to 0.
        end_dim : int, optional
            The end dimension to flatten. Defaults to -1.

        Returns
        -------
        BoundingBoxes
            The flattened boxes.
        """

        return self.__class__(
            tensor=self.tensor.flatten(start_dim=start_dim, end_dim=end_dim),
            images_size=self.images_size.flatten(start_dim=start_dim, end_dim=end_dim),
            format=self.format,
            normalized=self.normalized,
        )

    # -------------------------------------------------------------------------
    # Magic methods
    # -------------------------------------------------------------------------

    def __getitem__(
        self, index: int | slice | Tensor | tuple[int | slice | Tensor, ...]
    ) -> Self:
        """Returns the selected boxes.

        Parameters
        ----------
        index : int | slice | Tensor | tuple[int | slice | Tensor, ...]
            The index or indices to select.

        Returns
        -------
        BoundingBoxes
            The selected boxes.
        """

        return self.__class__(
            tensor=self.tensor[index],
            images_size=self.images_size[index],
            format=self.format,
            normalized=self.normalized,
        )


BoundingBoxes.__or__.__doc__ = BoundingBoxes.union.__doc__
BoundingBoxes.__and__.__doc__ = BoundingBoxes.intersection.__doc__
