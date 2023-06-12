##
##
##

from dataclasses import dataclass
from typing import Literal, Union, overload

import torch
from jaxtyping import Float, Int64
from torch import Tensor
from typing_extensions import Self

from visgator.utils.batch import Caption
from visgator.utils.bbox import BBoxes, BBoxFormat


@dataclass(frozen=True)
class CaptionEmbeddings:
    entities: Float[Tensor, "N D"]
    relations: Float[Tensor, "M D"]


@dataclass(frozen=True)
class DetectionResults:
    entities: Int64[Tensor, "N"]  # noqa: F821
    boxes: BBoxes


@dataclass(frozen=True)
class Graph:
    nodes: Float[Tensor, "N D"]
    edges: Float[Tensor, "M D"]
    edge_index: Int64[Tensor, "2 M"]

    @classmethod
    def new(
        cls,
        caption: Caption,
        embeddings: CaptionEmbeddings,
        detections: DetectionResults,
    ) -> Self:
        graph = caption.graph
        assert graph is not None

        nodes = embeddings.entities[detections.entities]

        edge_index_list = []
        edge_rel_index_list = []

        for idx, detection in enumerate(detections.entities):
            entity: int = detection.item()
            for connection in graph.connections(entity):
                if connection.end < entity:
                    continue

                device = detections.entities[0].device
                tmp = (
                    (detections.entities == connection.end)
                    .nonzero(as_tuple=True)[0]
                    .to(device)
                )
                indexes = torch.cat(
                    [
                        torch.tensor([idx])[None].expand(1, len(tmp)).to(device),
                        tmp[None],
                    ],
                    dim=0,
                )

                edge_index_list.append(indexes)
                edge_rel_index_list.extend([connection.relation] * len(tmp))

        if len(edge_index_list) == 0:
            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
                device=detections.entities.device,
            )
        else:
            edge_index = torch.cat(edge_index_list, dim=1)  # (2, M)

        edge_rel_index = torch.tensor(
            edge_rel_index_list,
            dtype=torch.long,
            device=embeddings.relations.device,
        )  # (M,)
        edges = embeddings.relations[edge_rel_index]

        return cls(nodes, edges, edge_index)


class NestedGraph:
    def __init__(
        self,
        nodes: Float[Tensor, "B N D"],
        edges: Float[Tensor, "B E D"],
        edge_index: Int64[Tensor, "2 BE"],
        sizes: list[tuple[int, int]],
    ) -> None:
        self._nodes = nodes
        self._edges = edges
        self._edge_index = edge_index
        self._sizes = sizes

    def new_like(
        self,
        nodes: Union[Float[Tensor, "BN D"], Float[Tensor, "B N D"]],
        edges: Union[Float[Tensor, "BE D"], Float[Tensor, "B E D"]],
    ) -> Self:
        """Creates a new NestedGraph with the same graph sizes as `self` but with
        the given nodes and edges."""

        batch = len(self)
        if nodes.ndim == 2:
            nodes = nodes.view(batch, -1, nodes.shape[1])
        if edges.ndim == 2:
            edges = edges.view(batch, -1, edges.shape[1])

        return self.__class__(nodes, edges, self._edge_index, self._sizes)

    @overload
    def nodes(self, batch: Literal[False]) -> Float[Tensor, "BN D"]:
        ...

    @overload
    def nodes(self, batch: Literal[True]) -> Float[Tensor, "B N D"]:
        ...

    def nodes(
        self, batch: bool
    ) -> Union[Float[Tensor, "BN D"], Float[Tensor, "B N D"]]:
        """Returns the nodes of the NestedGraph."""
        if batch:
            return self._nodes
        else:
            return self._nodes.flatten(0, 1)

    @overload
    def edges(self, batch: Literal[False]) -> Float[Tensor, "BE D"]:
        ...

    @overload
    def edges(self, batch: Literal[True]) -> Float[Tensor, "B E D"]:
        ...

    def edges(
        self, batch: bool
    ) -> Union[Float[Tensor, "BE D"], Float[Tensor, "B E D"]]:
        """Returns the edges of the NestedGraph."""
        if batch:
            return self._edges
        else:
            return self._edges.flatten(0, 1)

    @overload
    def edge_index(self, batch: Literal[False]) -> Int64[Tensor, "2 BE"]:
        ...

    @overload
    def edge_index(self, batch: Literal[True]) -> Int64[Tensor, "B 2 E"]:
        ...

    def edge_index(
        self, batch: bool
    ) -> Union[Int64[Tensor, "2 BE"], Int64[Tensor, "B 2 E"]]:
        """Returns the edge index of the NestedGraph."""
        if batch:
            raise NotImplementedError
        else:
            return self._edge_index

    @property
    def sizes(self) -> list[tuple[int, int]]:
        return self._sizes

    def to_graphs(self) -> list[Graph]:
        """Converts the NestedGraph to a list of Graphs."""

        graphs = []
        for i in range(len(self)):
            nodes = self._nodes[i, : self._sizes[i][0]]
            edges = self._edges[i, : self._sizes[i][1]]
            edge_index = self._edge_index[i, :, : self._sizes[i][1]]

            graphs.append(Graph(nodes, edges, edge_index))

        return graphs

    def __len__(self) -> int:
        return len(self._sizes)


def pad_sequences(
    detections: list[DetectionResults],
    graphs: list[Graph],
) -> tuple[BBoxes, NestedGraph]:
    if len(detections) != len(graphs):
        raise ValueError(
            f"The number of detections ({len(detections)}) must be equal "
            f"to the number of graphs ({len(graphs)})"
        )

    batch = len(detections)
    sizes = [(graph.nodes.shape[0], graph.edges.shape[0]) for graph in graphs]
    max_nodes = max([nodes for nodes, _ in sizes]) + 1
    max_edges = max([edges for _, edges in sizes])

    # We add one to the number of max_nodes to prevent the situation in which
    # the graph with max_nodes is not the graph with max_edges.
    # If we did not do this, when padding the edge_index we would add an edge
    # between two non existent padding nodes (graph.nodes.shape[0] == max_nodes)
    # causing an index out of bounds error.

    padded_boxes = detections[0].boxes.tensor.new_ones(batch * max_nodes, 4)
    images_size = detections[0].boxes.images_size.new_ones(batch * max_nodes, 2)

    nodes = graphs[0].nodes.new_zeros(batch, max_nodes, graphs[0].nodes.shape[1])
    edges = graphs[0].edges.new_zeros(batch, max_edges, graphs[0].edges.shape[1])
    edge_index = graphs[0].edge_index.new_empty(2, batch * max_edges)

    for i, (detection, graph) in enumerate(zip(detections, graphs)):
        nodes[i, : graph.nodes.shape[0]].copy_(graph.nodes)
        edges[i, : graph.edges.shape[0]].copy_(graph.edges)

        # Pad edge index
        start = i * max_edges
        middle = start + graph.edge_index.shape[1]
        end = start + max_edges

        edge_index[:, start:middle].copy_(graph.edge_index + i * max_nodes)
        edge_index[0, middle:end] = graph.nodes.shape[0] + i * max_nodes
        edge_index[1, middle:end] = graph.nodes.shape[0] + i * max_nodes

        # Pad boxes
        start = i * max_nodes
        boxes = detection.boxes.to_cxcywh().normalize()
        end = start + len(boxes)

        padded_boxes[start:end].copy_(boxes.tensor)
        images_size[start:end].copy_(boxes.images_size)

    boxes = BBoxes(padded_boxes, images_size, BBoxFormat.CXCYWH, True)

    return boxes, NestedGraph(nodes, edges, edge_index, sizes)
