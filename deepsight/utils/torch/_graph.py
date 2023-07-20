##
##
##

from dataclasses import dataclass
from typing import overload

import torch
from jaxtyping import Int64, Shaped
from torch import Tensor
from typing_extensions import Self

from deepsight.utils.protocols import Moveable

# -----------------------------------------------------------------------------
# Graph
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Graph(Moveable):
    """Dataclass to store graph data.

    Attributes
    ----------
    nodes : Shaped[Tensor, "N D"]
        Tensor of node features.
    edges : Shaped[Tensor, "E D"]
        Tensor of edge features.
    edge_indices : Int64[Tensor, "2 E"]
        Tensor of edge indices.
    """

    nodes: Shaped[Tensor, "N D"]
    edges: Shaped[Tensor, "E D"]
    edge_indices: Int64[Tensor, "2 E"]

    def __post_init__(self) -> None:
        if self.edges.shape[0] != self.edge_indices.shape[1]:
            raise ValueError(
                "Number of edges in `edges` and `edge_indices` must be equal."
            )

    def to(self, device: torch.device | str) -> Self:
        return self.__class__(
            nodes=self.nodes.to(device),
            edges=self.edges.to(device),
            edge_indices=self.edge_indices.to(device),
        )


# -----------------------------------------------------------------------------
# BatchedGraph
# -----------------------------------------------------------------------------


class BatchedGraphs(Moveable):
    """Class to store a batch of graphs data."""

    def __init__(
        self,
        nodes: Shaped[Tensor, "N D"] | Shaped[Tensor, "B N D"],
        edges: Shaped[Tensor, "E D"] | Shaped[Tensor, "B E D"],
        edge_indices: Int64[Tensor, "2 E"],
        sizes: list[tuple[int, int]],
    ):
        if nodes.ndim == 3:
            nodes_seq = [
                nodes[idx, :num_nodes] for idx, (num_nodes, _) in enumerate(sizes)
            ]
            nodes = torch.cat(nodes_seq, dim=0)
        self._nodes = nodes

        if edges.ndim == 3:
            edges_seq = [
                edges[idx, :num_edges] for idx, (_, num_edges) in enumerate(sizes)
            ]
            edges = torch.cat(edges_seq, dim=0)
        self._edges = edges

        self._edge_indices = edge_indices
        self._sizes = sizes

    @classmethod
    def from_list(cls, graphs: list[Graph]) -> Self:
        """Create a batched graph from a list of graphs.

        Parameters
        ----------
        graphs : list[Graph]
            List of graphs to batch.

        Returns
        -------
        BatchedGraphs
            Batched graph.
        """

        sizes = [(graph.nodes.shape[0], graph.edges.shape[0]) for graph in graphs]

        nodes = torch.cat([graph.nodes for graph in graphs], dim=0)
        edges = torch.cat([graph.edges for graph in graphs], dim=0)

        edge_indices = graphs[0].edge_indices.new_empty(2, edges.shape[0])

        edge_counter = 0
        node_counter = 0
        for i, (n_nodes, n_edges) in enumerate(sizes):
            edge_indices[:, edge_counter : edge_counter + n_edges] = (
                graphs[i].edge_indices + node_counter
            )
            edge_counter += n_edges
            node_counter += n_nodes

        return cls(
            nodes=nodes,
            edges=edges,
            edge_indices=edge_indices,
            sizes=sizes,
        )

    @overload
    def nodes(self, pad_value: None) -> Shaped[Tensor, "N D"]:
        """Returns the nodes of the batched graph as the nodes of a single graph."""

    @overload
    def nodes(self, pad_value: float) -> Shaped[Tensor, "B N D"]:
        """Returns the nodes of the batched graph as a batch of graphs with the same
        number of nodes. The nodes of each graph are padded with `pad_value`."""

    def nodes(
        self, pad_value: float | None
    ) -> Shaped[Tensor, "B N D"] | Shaped[Tensor, "N D"]:
        if pad_value is None:
            return self._nodes
        else:
            return torch.nn.utils.rnn.pad_sequence(
                self._nodes.split([n for n, _ in self._sizes]),
                batch_first=True,
                padding_value=pad_value,
            )

    @overload
    def edges(self, pad_value: None) -> Shaped[Tensor, "E D"]:
        """Returns the edges of the batched graph as the edges of a single disconnected
        graph."""

    @overload
    def edges(self, pad_value: float) -> Shaped[Tensor, "B E D"]:
        """Returns the edges of the batched graph as a batch of disconnected graphs with
        the same number of edges. The edges of each graph are padded with `pad_value`.
        """

    def edges(
        self, pad_value: float | None
    ) -> Shaped[Tensor, "B E D"] | Shaped[Tensor, "E D"]:
        if pad_value is None:
            return self._edges
        else:
            return torch.nn.utils.rnn.pad_sequence(
                self._edges.split([n for _, n in self._sizes]),
                batch_first=True,
                padding_value=pad_value,
            )

    @property
    def edge_indices(self) -> Int64[Tensor, "2 E"]:
        return self._edge_indices

    @property
    def sizes(self) -> list[tuple[int, int]]:
        return self._sizes

    def new_like(
        self,
        nodes: Shaped[Tensor, "N D"] | Shaped[Tensor, "B N D"] | None = None,
        edges: Shaped[Tensor, "E D"] | Shaped[Tensor, "B E D"] | None = None,
        clone: bool = False,
    ) -> Self:
        """Creates a new batched graph with the same sizes as the current one.

        Parameters
        ----------
        nodes : Shaped[Tensor, "N D"] | Shaped[Tensor, "B N D"] | None, optional
            Nodes of the new batched graph. If `None`, the nodes of the current batched
            graph are used. Defaults to `None`.
        edges : Shaped[Tensor, "E D"] | Shaped[Tensor, "B E D"] | None, optional
            Edges of the new batched graph. If `None`, the edges of the current batched
            graph are used. Defaults to `None`.
        clone : bool, optional
            If `True`, the nodes and edges are cloned. Defaults to `False`.

        Returns
        -------
        BatchedGraphs
            New batched graph.
        """

        if nodes is None:
            nodes = self._nodes.clone() if clone else self._nodes

        if edges is None:
            edges = self._edges.clone() if clone else self._edges

        nodes = self._nodes if nodes is None else nodes
        edges = self._edges if edges is None else edges

        return self.__class__(
            nodes=nodes,
            edges=edges,
            edge_indices=self._edge_indices if clone else self._edge_indices.clone(),
            sizes=self._sizes,
        )

    def to_list(self) -> list[Graph]:
        """Converts the batched graph to a list of graphs.

        Returns
        -------
        list[Graph]
            List of graphs.
        """

        graphs = []
        edge_counter = 0
        node_counter = 0
        for n_nodes, n_edges in self._sizes:
            graphs.append(
                Graph(
                    nodes=self._nodes[node_counter : node_counter + n_nodes],
                    edges=self._edges[edge_counter : edge_counter + n_edges],
                    edge_indices=self._edge_indices[
                        :, edge_counter : edge_counter + n_edges
                    ]
                    - node_counter,
                )
            )
            edge_counter += n_edges
            node_counter += n_nodes

        return graphs

    def to(self, device: torch.device | str) -> Self:
        return self.__class__(
            nodes=self._nodes.to(device),
            edges=self._edges.to(device),
            edge_indices=self._edge_indices.to(device),
            sizes=self._sizes,
        )

    # -------------------------------------------------------------------------
    # Magic methods
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Returns the number of graphs in the batch.

        Returns
        -------
        int
            Number of graphs in the batch.
        """
        return len(self.sizes)
