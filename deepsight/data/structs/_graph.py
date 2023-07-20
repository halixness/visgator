##
##
##

from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, overload

import rustworkx as rx
import serde
from typing_extensions import Self

# -------------------------------------------------------------------------- #
# Entity
# -------------------------------------------------------------------------- #


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True, slots=True)
class Entity:
    """An entity in a scene graph.

    Attributes
    ----------
    noun : str
        A single noun that describes the entity.
    phrase : str
        A phrase that describes the entity.
    """

    noun: str
    phrase: str

    @classmethod
    def from_dict(cls, data: dict[str, str | list[str]]) -> Self:
        """Deserializes an entity from a dictionary.

        Parameters
        ----------
        data : dict[str, str | list[str]]
            The serialized entity.

        Returns
        -------
        Self
            The deserialized entity.
        """
        return serde.from_dict(cls, data)

    def to_dict(self) -> dict[str, str | list[str]]:
        """Serializes the entity to a dictionary.

        Returns
        -------
        dict[str, str | list[str]]
            The serialized entity.
        """
        return serde.to_dict(self)


# -------------------------------------------------------------------------- #
# Triplet
# -------------------------------------------------------------------------- #

E = TypeVar("E")
P = TypeVar("P")


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True, slots=True)
class Triplet(Generic[E, P]):
    subject: E
    relation: P
    object: E

    @classmethod
    def from_dict(cls, data: dict[str, E | P]) -> Self:
        """Deserializes a triplet from a dictionary.

        Parameters
        ----------
        data : dict[str, E | P]
            The serialized triplet.

        Returns
        -------
        Self
            The deserialized triplet.
        """
        return serde.from_dict(cls, data)

    def to_dict(self) -> dict[str, E | P]:
        """Serializes the triplet to a dictionary.

        Returns
        -------
        dict[str, E | P]
            The serialized triplet.
        """
        return serde.to_dict(self)


# -------------------------------------------------------------------------- #
# SceneGraph
# -------------------------------------------------------------------------- #


class SceneGraph:
    def __init__(self, graph: rx.PyGraph) -> None:  # type: ignore
        self._graph: rx.PyGraph[Entity, str] = graph

    @classmethod
    def new(cls, entities: list[Entity], triplets: list[Triplet[int, str]]) -> Self:
        """Creates a new scene graph with the given entities and triplets.

        Parameters
        ----------
        entities : list[Entity]
            The entities in the scene graph.
        triplets : list[Relation[int, str]]
            The triplets in the scene graph.

        Returns
        -------
        Self
            The new scene graph.
        """

        graph: rx.PyGraph[Entity, str] = rx.PyGraph()
        graph.add_nodes_from(entities)
        graph.add_edges_from([(r.subject, r.object, r.relation) for r in triplets])

        return cls(graph)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserializes a scene graph from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The serialized scene graph.

        Returns
        -------
        Self
            The deserialized scene graph.
        """
        return cls.new(
            entities=[Entity.from_dict(entity) for entity in data["entities"]],
            triplets=[
                Triplet[int, str].from_dict(triplet) for triplet in data["triplets"]
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the scene graph to a dictionary.

        Returns
        -------
        dict[str, Any]
            The serialized scene graph.
        """

        return {
            "entities": [entity.to_dict() for entity in self.entities()],
            "triplets": [
                triplet.to_dict() for triplet in self.triplets(None, True, False)
            ],
        }

    def entities(self) -> list[Entity]:
        """Returns the entities in the scene graph.

        Returns
        -------
        list[Entity]
            The entities in the scene graph.
        """
        return self._graph.nodes()

    @overload
    def triplets(
        self,
        subject: int | None,
        return_entity_index: Literal[False],
        return_relation_index: Literal[False],
    ) -> list[Triplet[Entity, str]]:
        ...

    @overload
    def triplets(
        self,
        subject: int | None,
        return_entity_index: Literal[False],
        return_relation_index: Literal[True],
    ) -> list[Triplet[Entity, int]]:
        ...

    @overload
    def triplets(
        self,
        subject: int | None,
        return_entity_index: Literal[True],
        return_relation_index: Literal[False],
    ) -> list[Triplet[int, str]]:
        ...

    @overload
    def triplets(
        self,
        subject: int | None,
        return_entity_index: Literal[True],
        return_relation_index: Literal[True],
    ) -> list[Triplet[int, int]]:
        ...

    def triplets(
        self,
        subject: int | None = None,
        return_entity_index: bool = False,
        return_relation_index: bool = False,
    ) -> (
        list[Triplet[int, int]]
        | list[Triplet[int, str]]
        | list[Triplet[Entity, int]]
        | list[Triplet[Entity, str]]
    ):
        """Returns the triplets (subject, relation, object) in the scene graph.

        Parameters
        ----------
        subject : int | None, optional
            The index of the subject node for which to return the triplets.
            If None, all triplets are returned. Defaults to None.
        return_entity_index : bool, optional
            Whether to return the entity indices instead of the entities.
            Defaults to False.
        return_relation_index : bool, optional
            Whether to return the relation indices instead of the relations.

        Returns
        -------
        list[Triplet[int, int]] | list[Triplet[int, str]]
        | list[Triplet[Entity, int]] | list[Triplet[Entity, str]]
            The triplets in the scene graph.
        """

        indices: rx.EdgeIndices
        if subject is not None:
            indices = self._graph.incident_edges(subject)  # type: ignore
        else:
            indices = self._graph.edge_indices()

        triplets = []
        for index in indices:
            sub, obj = self._graph.get_edge_endpoints_by_index(index)  # type: ignore
            if subject is not None:
                if subject == obj:
                    sub, obj = obj, sub

            match (return_entity_index, return_relation_index):
                case (False, False):
                    sub = self._graph.get_node_data(sub)
                    obj = self._graph.get_node_data(obj)
                    rel = self._graph.get_edge_data_by_index(index)  # type: ignore
                    triplets.append(Triplet(subject=sub, relation=rel, object=obj))
                case (False, True):
                    sub = self._graph.get_node_data(sub)
                    obj = self._graph.get_node_data(obj)
                    triplets.append(Triplet(subject=sub, relation=index, object=obj))
                case (True, False):
                    rel = self._graph.get_edge_data_by_index(index)  # type: ignore
                    triplets.append(Triplet(subject=sub, relation=rel, object=obj))
                case (True, True):
                    triplets.append(Triplet(subject=sub, relation=index, object=obj))

        return triplets

    def node_connected_component(self, entity: int) -> Self:
        """Returns a subgraph containing the node and all nodes connected to it.

        Parameters
        ----------
        entity : int
            The index of the node to start the search from.

        Returns
        -------
        Self
            The subgraph containing the node and all nodes connected to it.
        """

        component = rx.node_connected_component(self._graph, entity)  # type: ignore
        nodes = list(component)
        return self.__class__(self._graph.subgraph(nodes))
