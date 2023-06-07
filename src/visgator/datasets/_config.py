##
##
##

import abc
from typing import Any

from typing_extensions import Self

from visgator.utils.misc import get_subclass, public_parent_module


class Config(abc.ABC):
    """Abstract base class for dataset configuration."""

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        """Deserializes a dictionary into a Config object."""
        module_path = str(cfg["module"])
        sub_cls = get_subclass(module_path, cls)
        return sub_cls.from_dict(cfg)

    @property
    def module(self) -> str:
        """The module path of the class."""
        return public_parent_module(self.__class__)

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serializes a Config object into a dictionary."""
