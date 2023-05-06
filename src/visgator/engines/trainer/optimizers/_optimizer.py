##
##
##

from __future__ import annotations

import importlib
from typing import Any, Iterable

from torch import Tensor, optim

from ._config import Config, TorchConfig


class Optimizer(optim.Optimizer):
    @staticmethod
    def from_config(
        config: Config, params: Iterable[Tensor | dict[str, Any]]
    ) -> Optimizer:
        if isinstance(config, TorchConfig):
            cls = getattr(optim, config.name)
            optimizer = cls(params, **config.args)
        else:
            child_module = config.name.lower()
            parent_module = ".".join(Optimizer.__module__.split(".")[:-1])
            module = importlib.import_module(f"{parent_module}.{child_module}")
            cls = getattr(module, "Optimizer")
            optimizer = cls(config, params)

        return optimizer  # type: ignore
