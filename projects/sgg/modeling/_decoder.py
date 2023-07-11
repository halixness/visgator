##
##
##

import torch
from jaxtyping import Float
from torch import Tensor, nn

from deepsight.data.structs import BoundingBoxes
from deepsight.modeling.layers import LayerScale, gnn, position
from deepsight.utils.torch import Batched3DTensors, BatchedGraphs

from ._config import DecoderConfig


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()

        self._num_heads = config.num_heads

        self.gaussian_heatmaps = position.GaussianHeatmaps()
        self.node_embeddings = position.SinusoidalBoxEmbeddings(
            config.hidden_dim, include_wh=True
        )
        self.edge_embeddings = position.SinusoidalPairwiseBoxEmbeddings(
            config.hidden_dim
        )

        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(
        self,
        features: Batched3DTensors,
        graphs: BatchedGraphs,
        boxes: BoundingBoxes,
    ) -> list[BatchedGraphs]:
        H, W = features.shape[-2:]

        edge_indices = graphs.edge_indices  # (2, E)
        first_boxes = boxes[edge_indices[0]]
        second_boxes = boxes[edge_indices[1]]
        union_boxes = first_boxes | second_boxes  # (E, 4)

        node_mask_list = []
        edge_mask_list = []
        for idx, (num_nodes, num_edges) in enumerate(graphs.sizes):
            node_mask_list.append(features.mask[idx, None].expand(num_nodes, -1, -1))
            edge_mask_list.append(features.mask[idx, None].expand(num_edges, -1, -1))

        nodes_mask = torch.cat(node_mask_list, dim=0)  # (N, H, W)
        edges_mask = torch.cat(edge_mask_list, dim=0)  # (E, H, W)

        node_heatmaps = self.gaussian_heatmaps(boxes, nodes_mask)  # (N, H, W)
        union_heatmaps = self.gaussian_heatmaps(union_boxes, edges_mask)  # (E, H, W)
        first_heatmaps = node_heatmaps[edge_indices[0]]  # (E, H, W)
        second_heatmaps = node_heatmaps[edge_indices[1]]  # (E, H, W)
        edge_heatmaps = torch.maximum(
            torch.maximum(first_heatmaps, second_heatmaps),
            union_heatmaps,
        )  # (E, H, W)

        node_heatmaps = node_heatmaps.flatten(1)  # (N, H * W)
        edge_heatmaps = edge_heatmaps.flatten(1)  # (E, H * W)
        heatmaps_graph = graphs.new_like(node_heatmaps, edge_heatmaps)

        heatmaps = torch.cat(
            [
                heatmaps_graph.nodes(pad_value=0.0),
                heatmaps_graph.edges(pad_value=0.0),
            ],
            dim=1,
        )  # (B, N + E, H * W)

        flattened_features = features.to_batched2d()  # (B, H * W, C)
        mask = flattened_features.mask[:, None].expand_as(heatmaps)
        attn_mask = heatmaps.masked_fill_(mask, -torch.inf)
        attn_mask = attn_mask.repeat(self._num_heads, 1, 1)  # (B * heads, N + E, H * W)

        node_embeddings = self.node_embeddings(boxes)  # (N, D)
        edge_embeddings = self.edge_embeddings(first_boxes, second_boxes)  # (E, D)
        embeddings_graph = graphs.new_like(node_embeddings, edge_embeddings)

        layer: DecoderLayer
        outputs = []
        for layer in self.layers:
            graphs = layer(
                graphs,
                embeddings_graph,
                flattened_features.tensor,
                attn_mask,
            )
            outputs.append(graphs)

        return outputs

    def __call__(
        self,
        features: Batched3DTensors,
        graphs: BatchedGraphs,
        boxes: BoundingBoxes,
    ) -> list[BatchedGraphs]:
        return super().__call__(features, graphs, boxes)  # type: ignore


class DecoderLayer(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()

        self.pre_cross_attn_layernorm = nn.LayerNorm(config.hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.post_cross_attn_layerscale = LayerScale(config.hidden_dim)

        self.pre_gat_layernorm = nn.LayerNorm(config.hidden_dim)
        self.gat = gnn.GATConv(
            config.hidden_dim,
            config.num_heads,
            bias=True,
            dropout=config.dropout,
        )
        self.post_gat_layerscale = LayerScale(config.hidden_dim)

        self.pre_ffn_layernorm = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
        )
        self.post_ffn_layerscale = LayerScale(config.hidden_dim)

    def _perform_cross_attention(
        self,
        graphs: BatchedGraphs,
        features: Float[Tensor, "B HW D"],
        attn_mask: Float[Tensor, "Bh (N+E) HW"],
    ) -> BatchedGraphs:
        nodes = graphs.nodes(pad_value=0.0)  # (B, N, D)
        edges = graphs.edges(pad_value=0.0)  # (B, E, D)
        # padded number of nodes and edges
        N, E = nodes.shape[1], edges.shape[1]

        queries = torch.cat([nodes, edges], dim=1)  # (B, N + E, D)
        queries, _ = self.cross_attn(
            queries,
            features,
            features,
            attn_mask=attn_mask,
            need_weights=False,
        )

        nodes, edges = torch.split(queries, [N, E], dim=1)
        return graphs.new_like(nodes, edges)

    def forward(
        self,
        graphs: BatchedGraphs,
        embeddings: BatchedGraphs,
        features: Float[Tensor, "B HW D"],
        attn_mask: Float[Tensor, "Bh (N+E) HW"],
    ) -> BatchedGraphs:
        # Perform cross-attention.
        nodes, edges = graphs.nodes(None), graphs.edges(None)
        N, E = nodes.shape[0], edges.shape[0]
        pre_cross_attn_queries = torch.cat([nodes, edges], dim=0)  # (N + E, D)
        queries = self.pre_cross_attn_layernorm(pre_cross_attn_queries)
        nodes, edges = torch.split(queries, [N, E], dim=0)
        graphs = graphs.new_like(nodes, edges)
        graphs = self._perform_cross_attention(graphs, features, attn_mask)
        nodes, edges = graphs.nodes(None), graphs.edges(None)
        queries = torch.cat([nodes, edges], dim=0)  # (N + E, D)
        post_cross_attn_queries = (
            pre_cross_attn_queries + self.post_cross_attn_layerscale(queries)
        )

        # Perform GAT
        pre_gat_queries = post_cross_attn_queries
        pre_gat_queries = self.pre_gat_layernorm(pre_gat_queries)
        nodes, edges = torch.split(pre_gat_queries, [N, E], dim=0)
        graphs = graphs.new_like(nodes, edges)
        graphs = self.gat(graphs, embeddings)
        nodes, edges = graphs.nodes(None), graphs.edges(None)
        queries = torch.cat([nodes, edges], dim=0)  # (N + E, D)
        post_gat_queries = pre_gat_queries + self.post_gat_layerscale(queries)

        # Perform FFN
        pre_ffn_queries = post_gat_queries
        queries = self.pre_ffn_layernorm(pre_gat_queries)
        queries = self.ffn(queries)
        post_fnn_queries = pre_ffn_queries + self.post_ffn_layerscale(queries)

        # Update graphs
        nodes, edges = torch.split(post_fnn_queries, [N, E], dim=0)
        return graphs.new_like(nodes, edges)

    def __call__(
        self,
        graphs: BatchedGraphs,
        embeddings: BatchedGraphs,
        features: Float[Tensor, "B HW D"],
        attn_mask: Float[Tensor, "Bh (N+E) HW"],
    ) -> BatchedGraphs:
        return super().__call__(graphs, embeddings, features, attn_mask)  # type: ignore
