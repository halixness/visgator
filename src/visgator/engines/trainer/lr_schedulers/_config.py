##
##
##

from __future__ import annotations

import abc
import enum
import importlib
from typing import Any

import torch


class Provider(enum.Enum):
    TORCH = enum.auto()
    CUSTOM = enum.auto()


class Config(abc.ABC):
    def __init__(self, cfg: dict[str, Any]) -> None:
        name = cfg.get("name", None)
        if name is None:
            raise ValueError("Missing 'name' field in lr_scheduler configuration.")
        else:
            self._name = str(name)

    @property
    def name(self) -> str:
        return self._name

    @abc.abstractproperty
    def provider(self) -> Provider:
        ...

    @staticmethod
    def from_dict(cfg: dict[str, Any]) -> Config:
        name = cfg.get("name", None)
        if name is None:
            raise ValueError("Missing 'name' field in lr_scheduler configuration.")

        name = str(name)
        if getattr(torch.optim.lr_scheduler, name, None) is not None:
            child_module = "torch"
        else:
            child_module = name.lower()

        parent_module = ".".join(Config.__module__.split(".")[:-1])
        module = importlib.import_module(f"{parent_module}.{child_module}")
        cls = getattr(module, "Config")

        return cls(cfg)  # type: ignore
