##
##
##

import abc
from dataclasses import dataclass
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from visgator.utils.bbox import BBoxes

_T = TypeVar("_T")


@dataclass
class LossInfo:
    name: str
    weight: float


class Criterion(Generic[_T], nn.Module, abc.ABC):
    """Criterion base class."""

    @abc.abstractproperty
    def losses(self) -> list[LossInfo]:
        ...

    @abc.abstractmethod
    def forward(self, output: _T, target: BBoxes) -> dict[str, Float[Tensor, ""]]:
        ...

    def __call__(self, output: _T, target: BBoxes) -> dict[str, Float[Tensor, ""]]:
        return super().__call__(output, target)  # type: ignore
