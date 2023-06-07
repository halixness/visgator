##
##
##

import abc
from typing import Generic, TypeVar

from torch import nn

from visgator.utils.bbox import BBoxes

_T = TypeVar("_T")


class PostProcessor(Generic[_T], nn.Module, abc.ABC):
    """PostProcessor base class."""

    @abc.abstractmethod
    def forward(self, output: _T) -> BBoxes:
        ...

    def __call__(self, output: _T) -> BBoxes:
        return super().__call__(output)  # type: ignore
