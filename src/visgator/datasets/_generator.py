##
##
##

import abc

from typing_extensions import Self

from visgator.utils.factory import get_subclass

from ._config import Config


class Generator(abc.ABC):
    """Abstract base class for dataset preprocessing and generation."""

    @classmethod
    def from_config(cls, config: Config) -> Self:
        sub_cls = get_subclass(cls, config.name)
        return sub_cls.from_config(config)

    @abc.abstractmethod
    def generate(self) -> None:
        ...
