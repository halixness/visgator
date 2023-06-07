##
##
##

import abc
from typing import Any, Callable, Iterable, Optional

from torch.nn import Parameter
from typing_extensions import Self

from visgator.utils.misc import get_subclass

from ._config import Config


class Optimizer(abc.ABC):
    """Abstract base class for optimizers."""

    @classmethod
    def from_config(cls, config: Config, params: Iterable[Parameter]) -> Self:
        """Instantiates an optimizer from a configuration."""
        sub_cls = get_subclass(config.module, cls)
        return sub_cls.from_config(config, params)

    @abc.abstractproperty
    def name(self) -> str:
        """The name of the optimizer."""

    @abc.abstractproperty
    def param_groups(self) -> list[dict[str, Any]]:
        """Returns the parameter groups of the optimizer."""

    @abc.abstractmethod
    def step(self, closure: Optional[Callable[[], float]] = None) -> None:
        """Performs a single optimization step."""

    @abc.abstractmethod
    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zeroes the gradients of all parameters."""

    @abc.abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Returns the state of the optimizer as a `dict`."""

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads the optimizer state."""
