##
##
##

from dataclasses import dataclass
from typing import Any

import rustworkx as rx
import serde
from typing_extensions import Self


@serde.serde
@dataclass(frozen=True)
class Relation:
    subject: int
    predicate: str
    object: int

    def __str__(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return serde.from_dict(cls, data)


@dataclass(frozen=True)
class Connection:
    start: int
    end: int
    relation: int


class SceneGraph:
    def __init__(self, entities: list[str], relations: list[Relation]) -> None:
        self._graph = rx.PyGraph()
        self._graph.add_nodes_from(entities)
        self._graph.add_edges_from(
            [(r.subject, r.object, r.predicate) for r in relations]
        )

        self._connections = {
            (n1, n2): idx
            for idx, (start, end) in enumerate(self._graph.edge_list())
            for n1, n2 in [(start, end), (end, start)]
        }

    @property
    def entities(self) -> list[str]:
        """Returns a list of entities in the graph."""
        return list(self._graph.nodes())

    @property
    def relations(self) -> list[Relation]:
        """Retruns a list of relations in the graph."""
        return [
            Relation(subject, self._graph.get_edge_data(subject, object), object)
            for subject, object in self._graph.edge_list()
        ]

    def connections(self, entity: int) -> list[Connection]:
        """Returns a list of connections for the given entity."""
        return [
            Connection(entity, neighbor, self._connections[(entity, neighbor)])
            for neighbor in self._graph.neighbors(entity)
        ]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Creates a SceneGraph from a dictionary."""
        return cls(
            entities=data["entities"],
            relations=[Relation.from_dict(r) for r in data["relations"]],
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the SceneGraph."""
        return {
            "entities": list(self._graph.nodes()),
            "relations": [rel.to_dict() for rel in self.relations],
        }
