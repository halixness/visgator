##
##
##

from __future__ import annotations

import abc
from typing import Any

from typing_extensions import Self

from visgator.utils.factory import get_subclass


class Config(abc.ABC):
    """Abstract base class for dataset configuration."""

    @property
    def name(self) -> str:
        return self.__module__.split(".")[-2]

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        name = cfg.get("name", None)
        if name is None:
            raise ValueError("Missing 'name' field in dataset configuration.")

        sub_cls = get_subclass(cls, str(name))
        return sub_cls.from_dict(cfg)
