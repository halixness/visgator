##
##
##

from __future__ import annotations

import abc
from typing import Any

from typing_extensions import Self

from visgator.optimizers import Optimizer
from visgator.utils.misc import get_subclass

from ._config import Config


class LRScheduler(abc.ABC):
    @classmethod
    def from_config(cls, config: Config, optimizer: Optimizer) -> Self:
        """Instantiates a LRScheduler from a configuration."""
        sub_cls = get_subclass(config.module, cls)
        return sub_cls.from_config(config, optimizer)

    @abc.abstractproperty
    def name(self) -> str:
        """The name of the LRScheduler."""

    @abc.abstractmethod
    def step_after_epoch(self) -> None:
        """Performs a step after each epoch."""
        ...

    @abc.abstractmethod
    def step_after_batch(self) -> None:
        """Performs a step after each batch."""
        ...

    @abc.abstractproperty
    def last_lr(self) -> float:
        """Returns the last learning rate."""
        ...

    @abc.abstractmethod
    def state_dict(self) -> dict[str, Any]:
        ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        ...
