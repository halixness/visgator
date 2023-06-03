##
##
##

import enum
from typing import Iterator, Union, overload

import torch
from jaxtyping import Float, Int
from torch import Size, Tensor
from typing_extensions import Self

from . import ops


class BBoxFormat(enum.Enum):
    """An enum representing the bounding box format."""

    XYXY = "xyxy"
    XYWH = "xywh"
    CXCYWH = "cxcywh"

    def __str__(self) -> str:
        return self.value


# This is essentially a copy of BBoxes to make it easier to work with
# a single bounding box.
class BBox:
    def __init__(
        self,
        box: Union[Float[Tensor, "4"], tuple[float, float, float, float]],
        image_size: Union[tuple[int, int], Int[Tensor, "2"], Size],
        format: BBoxFormat,
        normalized: bool,
    ) -> None:
        if isinstance(box, tuple):
            box = torch.tensor(box)
        self._box = box[None]  # (1, 4)

        if isinstance(image_size, tuple):
            image_size = torch.tensor(image_size, device=box.device)
        elif isinstance(image_size, Size):
            image_size = torch.tensor(image_size, device=box.device)
        self._image_size = image_size[None]  # (1, 2)

        self._format = format
        self._normalized = normalized

    def to_xyxy(self) -> Self:
        """Converts the bounding box to the format (x1, y1, x2, y2)."""
        match self._format:
            case BBoxFormat.XYXY:
                return self
            case BBoxFormat.XYWH:
                box = ops.from_xywh_to_xyxy(self._box)
            case BBoxFormat.CXCYWH:
                box = ops.from_cxcywh_to_xyxy(self._box)
            case _:
                raise ValueError(f"Unknown bounding box format: {self._format}")

        return self.__class__(
            box,
            self._image_size,
            BBoxFormat.XYXY,
            self._normalized,
        )

    def to_xywh(self) -> Self:
        """Converts the bounding box to the format (x, y, w, h)."""
        match self._format:
            case BBoxFormat.XYXY:
                box = ops.from_xyxy_to_xywh(self._box)
            case BBoxFormat.XYWH:
                return self
            case BBoxFormat.CXCYWH:
                box = ops.from_cxcywh_to_xywh(self._box)
            case _:
                raise ValueError(f"Unknown bounding box format: {self._format}")

        return self.__class__(
            box,
            self._image_size,
            BBoxFormat.XYWH,
            self._normalized,
        )

    def to_cxcywh(self) -> Self:
        """Converts the bounding box to the format (cx, cy, w, h)."""
        match self._format:
            case BBoxFormat.XYXY:
                box = ops.from_xyxy_to_cxcywh(self._box)
            case BBoxFormat.XYWH:
                box = ops.from_xywh_to_cxcywh(self._box)
            case BBoxFormat.CXCYWH:
                return self
            case _:
                raise ValueError(f"Unknown bounding box format: {self._format}")

        return self.__class__(
            box,
            self._image_size,
            BBoxFormat.CXCYWH,
            self._normalized,
        )

    def normalize(self) -> Self:
        """Normalizes the bounding box to [0, 1] range."""
        if self._normalized:
            return self

        box = ops.normalize(self._box, self._image_size)
        return self.__class__(
            box,
            self._image_size,
            self._format,
            True,
        )

    def denormalize(self) -> Self:
        """Denormalizes the bounding box from [0, 1] range to image range."""
        if not self._normalized:
            return self

        box = ops.denormalize(self._box, self._image_size)
        return self.__class__(
            box,
            self._image_size,
            self._format,
            False,
        )

    def to(self, device: torch.device) -> Self:
        """Moves the bounding box to the given device."""
        return self.__class__(
            self._box.to(device),
            self._image_size.to(device),
            self._format,
            self._normalized,
        )

    @property
    def device(self) -> torch.device:
        """Returns the device of the bounding box."""
        return self._box.device

    @property
    def format(self) -> BBoxFormat:
        """Returns the format of the bounding box."""
        return self._format

    @property
    def normalized(self) -> bool:
        """Returns whether the bounding box is normalized."""
        return self._normalized

    @property
    def image_size(self) -> tuple[int, int]:
        """Returns the image size of the bounding box."""
        return (
            self._image_size[0, 0].item(),
            self._image_size[0, 1].item(),
        )  # type: ignore

    @property
    def tensor(self) -> Float[Tensor, "4"]:
        """Returns the bounding box as a tensor."""
        return self._box


# -----------------------------------------------------------------------------
# BBoxes
# -----------------------------------------------------------------------------


class BBoxes:
    """A collection of bounding boxes."""

    def __init__(
        self,
        boxes: Float[Tensor, "N 4"],
        images_size: Union[
            tuple[int, int],
            Int[Tensor, "2"],
            list[tuple[int, int]],
            Int[Tensor, "N 2"],
        ],
        format: BBoxFormat,
        normalized: bool,
    ) -> None:
        self._boxes = boxes

        if isinstance(images_size, list):
            images_size = torch.tensor(images_size, device=boxes.device)
            if len(images_size) != len(boxes):
                raise ValueError(
                    f"Expected {len(boxes)} images sizes, got {len(images_size)}."
                )
        elif isinstance(images_size, tuple):
            images_size = (
                torch.tensor(images_size, device=boxes.device)
                .unsqueeze(0)
                .expand(len(boxes), 2)
            )
        elif images_size.ndim == 1:
            images_size = images_size.unsqueeze(0).expand(len(boxes), 2)
        self._images_size = images_size

        self._format = format
        self._normalized = normalized

    @property
    def device(self) -> torch.device:
        """Returns the device of the bounding boxes."""
        return self._boxes.device

    @property
    def format(self) -> BBoxFormat:
        """Returns the format of the bounding boxes."""
        return self._format

    @property
    def normalized(self) -> bool:
        """Returns whether the bounding boxes are normalized."""
        return self._normalized

    @property
    def images_size(self) -> Int[Tensor, "N 2"]:
        """Returns the image sizes of the bounding boxes."""
        return self._images_size

    @property
    def tensor(self) -> Float[Tensor, "N 4"]:
        """Returns the bounding boxes as a tensor."""
        return self._boxes

    @classmethod
    def from_bboxes(cls, bboxes: list[BBox]) -> Self:
        """Creates a collection of bounding boxes from a list of bounding boxes."""
        format = bboxes[0].format
        normalized = bboxes[0].normalized

        if any(bbox.format != format for bbox in bboxes):
            raise ValueError("All bounding boxes must have the same format.")

        if any(bbox.normalized != normalized for bbox in bboxes):
            raise ValueError("All bounding boxes must have the same normalization.")

        boxes = torch.stack([bbox.tensor for bbox in bboxes], dim=0)
        images_size = [bbox.image_size for bbox in bboxes]

        return cls(boxes, images_size, format, normalized)

    def to_xyxy(self) -> Self:
        """Converts the bounding boxes to the format (x1, y1, x2, y2)."""
        match self._format:
            case BBoxFormat.XYXY:
                return self
            case BBoxFormat.XYWH:
                bboxes = ops.from_xywh_to_xyxy(self._boxes)
            case BBoxFormat.CXCYWH:
                bboxes = ops.from_cxcywh_to_xyxy(self._boxes)
            case _:
                raise ValueError(f"Unknown bounding box format: {self._format}")

        return self.__class__(
            bboxes,
            self._images_size,
            BBoxFormat.XYXY,
            self._normalized,
        )

    def to_xywh(self) -> Self:
        """Converts the bounding boxes to the format (x, y, w, h)."""
        match self._format:
            case BBoxFormat.XYXY:
                bboxes = ops.from_xyxy_to_xywh(self._boxes)
            case BBoxFormat.XYWH:
                return self
            case BBoxFormat.CXCYWH:
                bboxes = ops.from_cxcywh_to_xywh(self._boxes)
            case _:
                raise ValueError(f"Unknown bounding box format: {self._format}")

        return self.__class__(
            bboxes,
            self._images_size,
            BBoxFormat.XYWH,
            self._normalized,
        )

    def to_cxcywh(self) -> Self:
        """Converts the bounding boxes to the format (cx, cy, w, h)."""
        match self._format:
            case BBoxFormat.XYXY:
                bboxes = ops.from_xyxy_to_cxcywh(self._boxes)
            case BBoxFormat.XYWH:
                bboxes = ops.from_xywh_to_cxcywh(self._boxes)
            case BBoxFormat.CXCYWH:
                return self
            case _:
                raise ValueError(f"Unknown bounding box format: {self._format}")

        return self.__class__(
            bboxes,
            self._images_size,
            BBoxFormat.CXCYWH,
            self._normalized,
        )

    def normalize(self) -> Self:
        """Normalizes the bounding boxes to [0, 1] range."""
        if self._normalized:
            return self

        bboxes = ops.normalize(self._boxes, self._images_size)
        return self.__class__(
            bboxes,
            self._images_size,
            self._format,
            True,
        )

    def denormalize(self) -> Self:
        """Denormalizes the bounding boxes from [0, 1] range to image range."""
        if not self._normalized:
            return self

        bboxes = ops.denormalize(self._boxes, self._images_size)
        return self.__class__(
            bboxes,
            self._images_size,
            self._format,
            False,
        )

    def to(self, device: torch.device) -> Self:
        """Moves the bounding boxes to the given device."""
        return self.__class__(
            self._boxes.to(device),
            self._images_size.to(device),
            self._format,
            self._normalized,
        )

    def area(self) -> Float[Tensor, "N"]:  # noqa: F821
        """Returns the area of the bounding boxes."""
        match self._format:
            case BBoxFormat.XYXY:
                width = self._boxes[:, 2] - self._boxes[:, 0]
                height = self._boxes[:, 3] - self._boxes[:, 1]
                return width * height
            case BBoxFormat.XYWH:
                return self._boxes[:, 2] * self._boxes[:, 3]
            case BBoxFormat.CXCYWH:
                return self._boxes[:, 2] * self._boxes[:, 3]
            case _:
                raise ValueError(f"Unknown bounding box format: {self._format}")

    def union(self, other: Self) -> Self:
        """Returns the union of the bounding boxes."""

        if self._images_size != other._images_size:
            raise ValueError("Bounding boxes must have the same image size.")

        if len(self) != len(other):
            raise ValueError("Bounding boxes must have the same length.")

        if self._normalized != other._normalized:
            boxes1 = self.to_xyxy().normalize()._boxes
            boxes2 = other.to_xyxy().normalize()._boxes
        else:
            boxes1 = self.to_xyxy()._boxes
            boxes2 = other.to_xyxy()._boxes

        union = ops.union_box_pairwise(boxes1, boxes2)

        return self.__class__(
            union,
            self._images_size,
            BBoxFormat.XYXY,
            self._normalized,
        )

    def __len__(self) -> int:
        return len(self._boxes)

    @overload
    def __getitem__(self, index: int) -> BBox:
        ...

    @overload
    def __getitem__(self, index: Union[slice, Tensor]) -> Self:
        ...

    def __getitem__(self, index: Union[int, slice, Tensor]) -> Union[BBox, Self]:
        if isinstance(index, int):
            return BBox(
                self._boxes[index],
                self._images_size[index],
                self._format,
                self._normalized,
            )
        elif isinstance(index, slice):
            return self.__class__(
                self._boxes[index],
                self._images_size[index],
                self._format,
                self._normalized,
            )
        elif isinstance(index, Tensor):
            return self.__class__(
                self._boxes[index],
                self._images_size[index],
                self._format,
                self._normalized,
            )
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __iter__(self) -> Iterator[BBox]:
        for i in range(len(self)):
            yield self[i]
