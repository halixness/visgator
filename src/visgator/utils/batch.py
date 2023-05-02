##
##
##

import typing
from dataclasses import dataclass
from typing import Iterator, Self

import torch
from jaxtyping import Float
from torch import Tensor


@dataclass(frozen=True)
class BatchSample:
    image: Float[Tensor, "3 H W"]
    sentence: str

    def to(self, device: torch.device) -> Self:
        """Moves the sample to the given device."""
        return self.__class__(self.image.to(device), self.sentence)


@dataclass(frozen=True)
class Batch:
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
