##
##
##

import re
from typing import Any, Callable, Iterable, Optional

import torch
from torch.nn import Parameter
from typing_extensions import Self

from deepsight.optimizers import Optimizer as _Optimizer

from ._config import Config


class Optimizer(_Optimizer):
    def __init__(
        self,
        config: Config,
        named_params: Iterable[tuple[str, Parameter]],
    ) -> None:
        cls = getattr(torch.optim, config.name)
        optimizer: torch.optim.Optimizer

        self.group_names = [g.regex for g in config.groups]

        param_groups: list[dict[str, Any]] = []
        for pg in config.groups:
            param_groups.append({"params": [], **pg.args})

        for name, param in named_params:
            freeze = False
            for freeze_regex in config.freeze:
                if re.match(freeze_regex, name):
                    param.requires_grad_(False)
                    freeze = True
                    break

            if freeze:
                continue

            for idx, group in enumerate(config.groups):
                if re.match(group.regex, name):
                    param_groups[idx]["params"].append(param)
                    break
            else:
                raise ValueError(f"Parameter {name} does not match any group.")

        optimizer = cls(param_groups)

        self._optimizer = optimizer

    @classmethod
    def new(
        cls,
        config: Config,  # type: ignore
        params: Iterable[tuple[str, Parameter]],
    ) -> Self:
        return cls(config, params)

    @property
    def name(self) -> str:
        return self._optimizer.__class__.__name__

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self._optimizer.param_groups

    def get_param_groups_names(self) -> list[str]:
        return self.group_names

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
