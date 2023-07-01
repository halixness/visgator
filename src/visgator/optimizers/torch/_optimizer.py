##
##
##

from typing import Any, Callable, Iterable, Optional

import torch
from torch.nn import Parameter
from typing_extensions import Self

from visgator.optimizers import Optimizer as _Optimizer

from ._config import Config


class Optimizer(_Optimizer):
    def __init__(self, config: Config, params: Iterable[Parameter]) -> None:
        cls = getattr(torch.optim, config.name)
        self._optimizer: torch.optim.Optimizer = cls(params, **config.args)

    @classmethod
    def from_config(
        cls,
        config: Config,  # type: ignore
        params: Iterable[Parameter],
    ) -> Self:
        return cls(config, params)

    @property
    def name(self) -> str:
        return self._optimizer.__class__.__name__

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self._optimizer.param_groups

    def step(self, closure: Optional[Callable[[], float]] = None) -> None:
        self._optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._optimizer.zero_grad(set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._optimizer.load_state_dict(state_dict)

    def __getattribute__(self, __name: str) -> Any:
        if __name == "__class__":
            return self._optimizer.__class__

        try:
            return super().__getattribute__(__name)
        except AttributeError:
            return getattr(self._optimizer, __name)
