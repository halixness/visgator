##
##
##

import abc

from typing_extensions import Self

from visgator.utils.misc import get_subclass

from ._config import Config


class Generator(abc.ABC):
    """Abstract base class for dataset preprocessing and generation."""

    @classmethod
    @abc.abstractmethod
    def new(cls, config: Config) -> Self:
        """Instantiates a generator from a configuration."""
        sub_cls = get_subclass(config.module, cls)
        return sub_cls.new(config)

    @abc.abstractmethod
    def generate(self) -> None:
        ...
