##
##
##

from dataclasses import dataclass
from typing import Any

import rustworkx as rx
import serde
from typing_extensions import Self


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class Entity:
    """Represents an entity in the scene graph.

    Attributes
    ----------
    span : str
        The span of the entity in the original sentence.
        For example, "the red ball" or "the table" in "the red ball is on the table".
    head : str
        The head of the entity in the original sentence.
        For example, "ball" or "table" in "the red ball is on the table".
    """

    span: str
    head: str

    def to_dict(self) -> dict[str, str]:
        return serde.to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Self:
        return serde.from_dict(cls, data)


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class Relation:
    """Represents a relation in the scene graph.

    Attributes
    ----------
    subject : int
        The index of the subject entity in the scene graph.
    predicate : str
        The predicate of the relation.
        For example, "is on" in "the red ball is on the table".
    object : int
        The index of the object entity in the scene graph.
    """

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
    """Represents a scene graph.

    Attributes
    ----------
    entities : list[Entity]
        A list of entities in the scene graph.
    relations : list[Relation]
        A list of relations in the scene graph.
    """

    def __init__(self, entities: list[Entity], relations: list[Relation]) -> None:
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
    def entities(self) -> list[Entity]:
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
            entities=[Entity.from_dict(e) for e in data["entities"]],
            relations=[Relation.from_dict(r) for r in data["relations"]],
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the SceneGraph."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [rel.to_dict() for rel in self.relations],
        }
