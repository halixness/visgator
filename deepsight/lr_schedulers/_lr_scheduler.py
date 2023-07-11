##
##
##

import abc
import importlib
from typing import Any

from typing_extensions import Self

from deepsight.optimizers import Optimizer

from ._config import Config


class LRScheduler(abc.ABC):
    @classmethod
    def new(
        cls,
        config: Config,
        optimizer: Optimizer,
        num_epoch: int,
        steps_per_epoch: int,
    ) -> Self:
        """Creates a new learning rate scheduler given a `config` and `optimizer`.

        Calling this method on a subclass of `LRScheduler` will return an instance of
        that subclass. Calling it on `LRScheduler` will return an instance of a subclass
        of `LRScheduler` depending on the `module` field of the `config`. The `module`
        field should be a valid module path to a subclass of `LRScheduler`.

        Parameters
        ----------
        config : Config
            The learning rate scheduler config.
        optimizer : Optimizer
            The optimizer.
        num_epoch : int
            The number of epochs to train for.
        steps_per_epoch : int
            The number of steps per epoch.

        Returns
        -------
        LRScheduler
            The learning rate scheduler instance.
        """

        module = importlib.import_module(config.module)
        sub_cls = getattr(module, cls.__name__)
        return sub_cls.new(  # type: ignore
            config,
            optimizer,
            num_epoch,
            steps_per_epoch,
        )

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the LRScheduler."""

    @abc.abstractmethod
    def step_after_epoch(self) -> None:
        """Performs a step after each epoch."""

    @abc.abstractmethod
    def step_after_batch(self) -> None:
        """Performs a step after each batch."""

    @abc.abstractmethod
    def get_last_lr(self) -> list[float]:
        """Returns the last learning rate for each parameter group."""
        ...

    @abc.abstractmethod
    def state_dict(self) -> dict[str, Any]:
        ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        ...
