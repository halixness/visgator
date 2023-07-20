##
##
##

import abc
import importlib
from typing import Any, Callable, Iterable, Optional

from torch.nn import Parameter
from typing_extensions import Self

from ._config import Config


class Optimizer(abc.ABC):
    """Abstract base class for optimizers."""

    @classmethod
    @abc.abstractmethod
    def new(cls, config: Config, params: Iterable[tuple[str, Parameter]]) -> Self:
        """Creates a new optimizer given a `config` and `params`.

        Calling this method on a subclass of `Optimizer` will return an instance of
        that subclass. Calling it on `Optimizer` will return an instance of a subclass
        of `Optimizer` depending on the `module` field of the `config`. The `module`
        field should be a valid module path to a subclass of `Optimizer`.

        Parameters
        ----------
        config : Config
            The optimizer config.
        params : Iterable[Parameter]
            The parameters to optimize.

        Returns
        -------
        Optimizer
            The optimizer instance.
        """

        module = importlib.import_module(config.module)
        sub_cls = getattr(module, cls.__name__)
        return sub_cls.new(config, params)  # type: ignore

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the optimizer."""

    @property
    @abc.abstractmethod
    def param_groups(self) -> list[dict[str, Any]]:
        """Returns the parameter groups of the optimizer."""

    @abc.abstractmethod
    def get_param_groups_names(self) -> list[str]:
        """Returns the names of the parameter groups of the optimizer."""

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
