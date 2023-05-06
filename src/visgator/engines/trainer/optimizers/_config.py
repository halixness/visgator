##
##
##

from __future__ import annotations

import importlib
from typing import Any

import torch


class Config:
    def __init__(self, cfg: dict[str, Any]) -> None:
        name = cfg.get("name", None)
        if name is None:
            raise ValueError("Missing 'name' field in optimizer configuration.")
        else:
            self._name = str(name)

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def from_dict(cfg: dict[str, Any]) -> Config:
        name = cfg.get("name", None)
        if name is None:
            raise ValueError("Missing 'name' field in optimizer configuration.")

        name = str(name)

        config: Config
        if getattr(torch.optim, name, None) is not None:
            config = TorchConfig(cfg)
        else:
            child_module = name.lower()
            parent_module = ".".join(Config.__module__.split(".")[:-1])
            module = importlib.import_module(f"{parent_module}.{child_module}")
            cls = getattr(module, "Config")
            config = cls(cfg)

        return config


class TorchConfig(Config):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)

        self._args = dict(cfg.get("args", {}))

    @property
    def args(self) -> dict[str, Any]:
        return self._args
