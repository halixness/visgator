##
##
##

import enum
import typing
from typing import Iterator, Optional

import torch
from jaxtyping import Float
from torch import Tensor
from typing_extensions import Self

from . import _utils as utils


class BBoxFormat(enum.Enum):
    """An enum representing the bounding box format."""

    XYXY = "xyxy"
    XYWH = "xywh"
    CXCYWH = "cxcywh"
    XYXYN = "xyxyn"
    XYWHN = "xywhn"
    CXCYWHN = "cxcywhn"

    def __str__(self) -> str:
        return self.value


class BBox:
    """A bounding box."""

    def __init__(
        self,
        bbox: Float[Tensor, "4"],
        image_size: tuple[int, int],
        format: BBoxFormat = BBoxFormat.XYXY,
    ):
        match format:
            case BBoxFormat.XYXY:
                self._bbox = bbox
            case BBoxFormat.XYWH:
                self._bbox = utils.xywh_to_xyxy(bbox)
            case BBoxFormat.CXCYWH:
                self._bbox = utils.cxcywh_to_xyxy(bbox)
            case BBoxFormat.XYXYN:
                self._bbox = utils.xyxyn_to_xyxy(bbox, image_size)
            case BBoxFormat.XYWHN:
                self._bbox = utils.xywhn_to_xyxy(bbox, image_size)
            case BBoxFormat.CXCYWHN:
                self._bbox = utils.cxcywhn_to_xyxy(bbox, image_size)
            case _:
                raise ValueError(f"Invalid bounding box format: {format}.")
        self._image_size = image_size

    @classmethod
    def from_tuple(
        cls,
        bbox: tuple[float, float, float, float],
        image_size: tuple[int, int],
        format: BBoxFormat = BBoxFormat.XYXY,
    ) -> Self:
        """Creates a bounding box from a tuple."""
        return cls(torch.tensor(bbox, dtype=torch.float32), image_size, format)

    @property
    def xyxy(self) -> Float[Tensor, "4"]:
        """Returns the bounding box in the format (x1, y1, x2, y2)."""
        return self._bbox

    @property
    def cxcywh(self) -> Float[Tensor, "4"]:
        """Returns the bounding box in the format (cx, cy, w, h)."""

        x1, y1, x2, y2 = self._bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        return self._bbox.new_tensor([cx, cy, w, h])

    @property
    def xyxyn(self) -> Float[Tensor, "4"]:
        """Returns the bounding box in the format (x1, y1, x2, y2) normalized."""
        h, w = self._image_size
        return self.xyxy / self._bbox.new_tensor([w, h, w, h])

    @property
    def cxcywhn(self) -> Float[Tensor, "4"]:
        """Returns the bounding box in the format (cx, cy, w, h) normalized."""
        h, w = self._image_size
        return self.cxcywh / self._bbox.new_tensor([w, h, w, h])

    @property
    def device(self) -> torch.device:
        """Returns the device of the bounding box."""
        return self._bbox.device

    def to(self, device: torch.device) -> Self:
        """Moves the bounding box to the given device."""
        return self.__class__(self._bbox.to(device), self._image_size)


class BBoxes:
    """A collection of bounding boxes."""

    def __init__(self, bboxes: Optional[list[BBox]] = None):
        if bboxes is None or len(bboxes) == 0:
            self._bboxes = []
        else:
            device = bboxes[0].device
            for bbox in bboxes:
                if bbox.device != device:
                    raise ValueError(
                        "All bounding boxes must be on the same device. "
                        f"Got {bbox.device} and {device}."
                    )
            self._bboxes = bboxes

    @property
    def xyxy(self) -> Float[Tensor, "N 4"]:
        """Returns the bounding boxes in the format (x1, y1, x2, y2)."""
        return torch.stack([bbox.xyxy for bbox in self._bboxes], dim=0)

    @property
    def cxcywh(self) -> Float[Tensor, "N 4"]:
        """Returns the bounding boxes in the format (cx, cy, w, h)."""
        return torch.stack([bbox.cxcywh for bbox in self._bboxes], dim=0)

    @property
    def xyxyn(self) -> Float[Tensor, "N 4"]:
        """Returns the bounding boxes in the format (x1, y1, x2, y2) normalized."""
        return torch.stack([bbox.xyxyn for bbox in self._bboxes], dim=0)

    @property
    def cxcywhn(self) -> Float[Tensor, "N 4"]:
        """Returns the bounding boxes in the format (cx, cy, w, h) normalized."""
        return torch.stack([bbox.cxcywhn for bbox in self._bboxes], dim=0)

    def append(self, bbox: BBox) -> None:
        """Appends a bounding box to the collection."""
        if len(self._bboxes) > 0 and bbox.device != self._bboxes[0].device:
            raise ValueError(
                "All bounding boxes must be on the same device. "
                f"Got {bbox.device} and {self._bboxes[0].device}."
            )
        self._bboxes.append(bbox)

    @property
    def device(self) -> torch.device:
        """Returns the device of the bounding boxes."""
        return self._bboxes[0].device

    def to(self, device: torch.device) -> Self:
        """Moves the bounding boxes to the given device."""
        return self.__class__([bbox.to(device) for bbox in self._bboxes])

    @typing.overload
    def __getitem__(self, index: int) -> BBox:
        ...

    @typing.overload
    def __getitem__(self, index: slice) -> Self:
        ...

    def __getitem__(self, index: int | slice) -> BBox | Self:
        if isinstance(index, int):
            return self._bboxes[index]

        return self.__class__(self._bboxes[index])

    def __len__(self) -> int:
        return len(self._bboxes)

    def __iter__(self) -> Iterator[BBox]:
        return iter(self._bboxes)
