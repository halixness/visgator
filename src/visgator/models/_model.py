##
##
##

from __future__ import annotations

import abc
from typing import Generic, Optional, TypeVar

from torch import nn

from visgator.utils.batch import Batch
from visgator.utils.factory import get_subclass

from ._config import Config
from ._criterion import Criterion
from ._postprocessor import PostProcessor

_T = TypeVar("_T")


class Model(nn.Module, Generic[_T], abc.ABC):
    """Model interface."""

    @classmethod
    def from_config(cls, config: Config) -> Model[_T]:
        sub_cls = get_subclass(cls, config.name)
        return sub_cls.from_config(config)

    @abc.abstractproperty
    def criterion(self) -> Optional[Criterion[_T]]:
        ...

    @abc.abstractproperty
    def postprocessor(self) -> PostProcessor[_T]:
        ...

    @abc.abstractmethod
    def forward(self, batch: Batch) -> _T:
        ...

    def __call__(self, batch: Batch) -> _T:
        return nn.Module.__call__(self, batch)  # type: ignore
