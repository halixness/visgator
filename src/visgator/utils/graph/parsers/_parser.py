##
##
##

import abc
from typing import Generator, Iterable

from typing_extensions import Self

from visgator.utils.graph import SceneGraph
from visgator.utils.misc import get_subclass

from ._config import Config


class Parser(abc.ABC):
    @classmethod
    def new(cls, config: Config) -> Self:
        """Instantiates a parser from a configuration."""
        sub_cls = get_subclass(config.module, cls)
        return sub_cls.new(config)

    @abc.abstractproperty
    def name(self) -> str:
        """The name of the parser."""

    @abc.abstractmethod
    def parse(self, sentences: Iterable[str]) -> Generator[SceneGraph, None, None]:
        """Parses a list of sentences into scene graphs."""
