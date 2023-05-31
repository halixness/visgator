##
##
##

import typing
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import serde
import torch
from jaxtyping import UInt8
from torch import Tensor
from typing_extensions import Self

from .graph import SceneGraph


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class Caption:
    """A caption with an optional scene graph."""

    sentence: str
    graph: Optional[SceneGraph] = serde.field(
        default=None,
        serializer=SceneGraph.from_dict,
        deserializer=SceneGraph.to_dict,
    )

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return serde.from_dict(cls, data)


@dataclass(frozen=True)
class BatchSample:
    """A batch sample with an image and a caption."""

    image: UInt8[Tensor, "3 H W"]
    caption: Caption

    def to(self, device: torch.device) -> Self:
        """Moves the sample to the given device."""
        return self.__class__(self.image.to(device), self.caption)


@dataclass(frozen=True)
class Batch:
    """A batch of samples."""

    samples: tuple[BatchSample, ...]

    def to(self, device: torch.device) -> Self:
        """Moves the batch to the given device."""
        return self.__class__(tuple(sample.to(device) for sample in self.samples))

    @typing.overload
    def __getitem__(self, index: int) -> BatchSample:
        ...

    @typing.overload
    def __getitem__(self, index: slice) -> Self:
        ...

    def __getitem__(self, index: int | slice) -> BatchSample | Self:
        if isinstance(index, int):
            return self.samples[index]

        return self.__class__(tuple(self.samples[index]))

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[BatchSample]:
        return iter(self.samples)
