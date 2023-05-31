##
##
##

from __future__ import annotations

import abc
from typing import Any

import torch
from typing_extensions import Self

from visgator.utils.factory import get_subclass


class Config(abc.ABC):
    @property
    def name(self) -> str:
        return self.__module__.split(".")[-2]

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        name = cfg.get("name", None)
        if name is None:
            raise ValueError("Missing 'name' field in lr_scheduler configuration.")

        name = str(name)
        if getattr(torch.optim.lr_scheduler, name, None) is not None:
            child_module = "torch"
        else:
            child_module = name

        sub_cls = get_subclass(cls, child_module)
        return sub_cls.from_dict(cfg)
