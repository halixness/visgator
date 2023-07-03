##
##
##

from dataclasses import dataclass
from typing import Any, Generator, Iterable

import sng_parser
from typing_extensions import Self

from visgator.utils.graph import Entity, Relation, SceneGraph
from visgator.utils.graph.parsers import Config as _Config
from visgator.utils.graph.parsers import Parser as _Parser


@dataclass(frozen=True, slots=True)
class Config(_Config):
    ...

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {}


class Parser(_Parser):
    def __init__(self, config: Config) -> None:
        super().__init__()

    @classmethod
    def new(cls, config: Config) -> Self:  # type: ignore
        return cls(config)

    @property
    def name(self) -> str:
        return "spaCy Scene Graph Parser"

    def parse(self, sentences: Iterable[str]) -> Generator[SceneGraph, None, None]:
        for sentence in sentences:
            graph = sng_parser.parse(sentence)

            entities = []
            for entity in graph["entities"]:
                entities.append(Entity(entity["span"], entity["head"]))

            relations = []
            for relation in graph["relations"]:
                relations.append(
                    Relation(
                        relation["subject"], relation["relation"], relation["object"]
                    )
                )

            yield SceneGraph(entities, relations)
