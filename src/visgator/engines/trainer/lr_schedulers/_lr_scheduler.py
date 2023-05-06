##
##
##

from __future__ import annotations

import abc
import importlib
from typing import Any

from visgator.engines.trainer.optimizers import Optimizer

from ._config import Config, Provider


class LRScheduler(abc.ABC):
    def __init__(self, config: Config, optimizer: Optimizer) -> None:
        ...

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

    @staticmethod
    def from_config(config: Config, optimizer: Optimizer) -> LRScheduler:
        match config.provider:
            case Provider.TORCH:
                child_module = "torch"
            case Provider.CUSTOM:
                child_module = config.name.lower()
            case _:
                raise ValueError(f"Unknown provider: {config.provider}.")

        parent_module = ".".join(LRScheduler.__module__.split(".")[:-1])
        module = importlib.import_module(f"{parent_module}.{child_module}")
        cls = getattr(module, "LRScheduler")

        return cls(config, optimizer)  # type: ignore
