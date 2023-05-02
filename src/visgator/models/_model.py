##
##
##

from __future__ import annotations

import abc
import importlib
from typing import Generic, TypeVar

from torch import nn

from visgator.utils.batch import Batch
from visgator.utils.bbox import BBoxes

from ._config import Config

_T = TypeVar("_T")


class Model(nn.Module, Generic[_T], abc.ABC):
    """Model interface."""

    def __init__(self, config: Config) -> None:
        super().__init__()

    def __call__(self, batch: Batch) -> _T:
        return nn.Module.__call__(self, batch)  # type: ignore

    @abc.abstractmethod
    def forward(self, batch: Batch) -> _T:
        ...

    @abc.abstractmethod
    def predict(self, output: _T) -> BBoxes:
        """Predicts bounding boxes from model output.

        Args:
            output (_T): the model output.

        Returns:
            BBoxes: the predicted bounding boxes.
        """
        ...

    @staticmethod
    def from_config(config: Config) -> Model[_T]:
        child_module = config.name.lower()
        parent_module = ".".join(Model.__module__.split(".")[:-1])
        module = importlib.import_module(f"{parent_module}.{child_module}")
        cls = getattr(module, "Model")

        return cls(config)  # type: ignore
