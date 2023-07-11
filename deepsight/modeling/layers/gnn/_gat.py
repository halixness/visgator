##
##
##

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add, scatter_softmax

from deepsight.utils.torch import BatchedGraphs


class GATConv(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        self.first_node_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.second_node_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.edge_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.attn_proj = nn.Parameter(torch.randn(1, num_heads, head_dim))

        self.node_out_proj = nn.Linear(embed_dim, embed_dim, bias)
        self.edge_out_proj = nn.Linear(embed_dim, embed_dim, bias)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        graphs: BatchedGraphs,
        embeddings: BatchedGraphs | None = None,
    ) -> BatchedGraphs:
        nodes = graphs.nodes(None)
        edges = graphs.edges(None)

        N, _ = nodes.shape
        E, _ = edges.shape
        H = self.num_heads

        query_nodes = nodes
        query_edges = edges

        if embeddings is not None:
            query_nodes = query_nodes + embeddings.nodes(None)
            query_edges = query_edges + embeddings.edges(None)

        first_node = self.first_node_proj(query_nodes)[graphs.edge_indices[0]]
        second_node = self.second_node_proj(query_nodes)[graphs.edge_indices[1]]
        query_edges = self.edge_proj(query_edges)

        hidden = first_node + second_node + query_edges
        hidden = F.gelu(hidden)

        hidden_head = hidden.view(E, H, -1)
        presoftmax_alpha = (hidden_head * self.attn_proj).sum(dim=-1)  # (E, H)
        alpha = scatter_softmax(presoftmax_alpha, graphs.edge_indices[0], dim=0)
        alpha = self.attn_dropout(alpha)

        new_edges = self.edge_out_proj(hidden)
        values = nodes[graphs.edge_indices[1]] + new_edges
        values = self.node_out_proj(values)
        values = values.view(E, H, -1)
        values = values * alpha.unsqueeze(-1)
        new_nodes = scatter_add(values, graphs.edge_indices[0], dim=0)
        new_nodes = new_nodes.view(N, -1)

        return graphs.new_like(nodes=new_nodes, edges=new_edges)

    def __call__(
        self,
        graphs: BatchedGraphs,
        embeddings: BatchedGraphs | None = None,
    ) -> BatchedGraphs:
        return super().__call__(graphs, embeddings)  # type: ignore
