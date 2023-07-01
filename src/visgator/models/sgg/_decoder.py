##
##
##

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torch_scatter import scatter_add, scatter_softmax

from visgator.utils.bbox import BBoxes
from visgator.utils.torch import Nested4DTensor

from ._config import DecoderConfig
from ._misc import NestedGraph
from ._position import (
    EntitySpatialEncodings,
    LogGaussianHeatmaps,
    RelationSpatialEncondings,
)


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()

        self._num_heads = config.num_heads
        self._hidden_dim = config.hidden_dim
        self._same_entity_edge = nn.Parameter(torch.randn(1, config.hidden_dim))

        # self._patch_encondings = PatchSpatialEncodings(config.hidden_dim)
        self._node_encodings = EntitySpatialEncodings(config.hidden_dim)
        self._edge_encodings = RelationSpatialEncondings(config.hidden_dim)
        self._gaussian_heatmaps = LogGaussianHeatmaps()

        self._layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(
        self,
        images: Nested4DTensor,
        graph: NestedGraph,
        boxes: BBoxes,
    ) -> NestedGraph:
        H, W = images.shape[2:]

        edge_index = graph.edge_index(False)  # (2 BE)
        boxes1 = boxes[edge_index[0]]  # (BE 4)
        boxes2 = boxes[edge_index[1]]  # (BE 4)
        union_boxes = boxes1.union(boxes2)  # (BE 4)

        heatmaps = self._gaussian_heatmaps(boxes, (H, W))  # (BN HW)
        union_heatmaps = self._gaussian_heatmaps(union_boxes, (H, W))
        heatmaps1 = heatmaps[edge_index[0]]  # (BE HW)
        heatmaps2 = heatmaps[edge_index[1]]  # (BE HW)
        edge_heatmaps = torch.maximum(
            torch.maximum(heatmaps1, heatmaps2),
            union_heatmaps,
        )  # (BE HW)

        # heatmaps = torch.log(heatmaps + 1e-8)  # (BN HW)
        # union_heatmaps = torch.log(union_heatmaps + 1e-8)  # (BE HW)

        node_heatmaps = heatmaps.view(len(graph), -1, H * W)  # (B N HW)
        edge_heatmaps = edge_heatmaps.view(len(graph), -1, H * W)  # (B E HW)
        heatmaps = torch.cat((node_heatmaps, edge_heatmaps), dim=1)  # (B (N+E) HW)

        flattened_images = images.flatten()  # (B HW D)
        masks = flattened_images.mask.unsqueeze(1).expand(-1, heatmaps.shape[1], -1)
        masks = heatmaps.masked_fill_(masks, -torch.inf)  # (B (N+E) HW)
        masks = masks.repeat(self._num_heads, 1, 1)  # (Bh (N+E) HW)

        node_encodings = self._node_encodings(boxes)  # (BN D)
        edge_encodings = self._edge_encodings(boxes1, boxes2)  # (BE D)
        # image_encodings = self._patch_encondings(images.mask)

        for block in self._layers:
            graph = block(
                graph,
                node_encodings,
                edge_encodings,
                flattened_images.tensor,
                masks,
            )

        return graph

    def __call__(
        self, images: Nested4DTensor, graph: NestedGraph, boxes: BBoxes
    ) -> NestedGraph:
        return super().__call__(images, graph, boxes)  # type: ignore


class DecoderLayer(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()

        # attention with images
        self._norm1 = nn.LayerNorm(config.hidden_dim)
        self._attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self._layerscale1 = LayerScale(config.hidden_dim, config.epsilon_layer_scale)

        # attention with graph
        self._norm2 = nn.LayerNorm(config.hidden_dim)
        self._gat = ModifiedGAT(config)
        self._layerscale2 = LayerScale(config.hidden_dim, config.epsilon_layer_scale)

        # feedforward
        self._norm3 = nn.LayerNorm(config.hidden_dim)
        self._ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
        )
        self._layerscale3 = LayerScale(config.hidden_dim, config.epsilon_layer_scale)

    def forward(
        self,
        graph: NestedGraph,
        node_encodings: Float[Tensor, "BN D"],
        edge_encodings: Float[Tensor, "BE D"],
        images: Float[Tensor, "B HW D"],
        mask: Float[Tensor, "Bh (N+E) HW"],
    ) -> NestedGraph:
        nodes, edges = graph.nodes(True), graph.edges(True)  # (B N D), (B E D)
        N = nodes.shape[1]
        ne = torch.cat((nodes, edges), dim=1)  # (B (N+E) D)

        # image attention
        ne1 = self._norm1(ne)  # (B (N+E) D)
        ne1, _ = self._attn(
            ne1,
            images,
            images,
            attn_mask=mask,
            need_weights=False,
        )  # (B (N+E) D)
        ne1 = ne + self._layerscale1(ne1)  # (B (N+E) D)

        # graph attention
        ne2 = self._norm2(ne1)  # (B (N+E) D)
        nodes, edges = ne2[:, :N], ne2[:, N:]  # (B N D), (B E D)
        graph = graph.new_like(nodes, edges)
        graph = self._gat(graph, node_encodings, edge_encodings)
        nodes, edges = graph.nodes(True), graph.edges(True)  # (B N D), (B E D)
        ne2 = torch.cat((nodes, edges), dim=1)  # (B (N+E) D)
        ne2 = ne1 + self._layerscale2(ne2)  # (B (N+E) D)

        # feedforward
        ne3 = self._norm3(ne2)  # (B (N+E) D)
        ne3 = self._ffn(ne3)  # (B (N+E) D)
        ne3 = ne2 + self._layerscale3(ne3)  # (B (N+E) D)

        nodes = ne3[:, :N]  # (B N D)
        edges = ne3[:, N:]  # (B E D)
        return graph.new_like(nodes, edges)


# Implementation based on GATv2Conv of PyTorch Geometric
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gatv2_conv.py
class ModifiedGAT(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()

        self._num_heads = config.num_heads
        head_dim = config.hidden_dim // config.num_heads

        self._first_node_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self._second_node_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self._edge_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self._attn_proj = nn.Parameter(torch.randn(1, config.num_heads, head_dim))

        self._edge_out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self._node_out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        self._dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        graph: NestedGraph,
        node_encondings: Float[Tensor, "BN D"],
        edge_encodings: Float[Tensor, "BE D"],
    ) -> NestedGraph:
        nodes = graph.nodes(False)  # (BN D)
        edges = graph.edges(False)  # (BE D)
        edge_index = graph.edge_index(False)  # (2 BE)

        BN, _ = nodes.shape
        BE, _ = edges.shape
        H = self._num_heads

        nodes = nodes + node_encondings  # (BN D)
        edges = edges + edge_encodings  # (BE D)

        first_nodes = self._first_node_proj(nodes)[edge_index[0]]  # (BE D)
        second_nodes = self._second_node_proj(nodes)[edge_index[0]]  # (BE D)
        edges = self._edge_proj(edges)  # (BE D)

        hidden = first_nodes + second_nodes + edges  # (BE D)
        hidden = F.gelu(hidden)  # (BE D)

        hidden_head = hidden.view(BE, H, -1)  # (BE H D/H)
        presoftmax_alpha = (hidden_head * self._attn_proj).sum(dim=-1)  # (BE H)
        alpha = scatter_softmax(presoftmax_alpha, edge_index[0], dim=0)  # (BE, H)
        alpha = self._dropout(alpha)  # (BE H)

        new_edges = self._edge_out_proj(hidden)  # (BE D)
        values = nodes[edge_index[1]] + new_edges  # (BE D)
        values = self._node_out_proj(values)  # (BE D)
        values = values.view(BE, H, -1)  # (BE H D/H)
        values = alpha.unsqueeze(-1) * values  # (BE H D/H)
        new_nodes = scatter_add(values, edge_index[0], dim=0)  # (BN H D/H)
        new_nodes = new_nodes.view(BN, -1)  # (BN D)

        return graph.new_like(new_nodes, new_edges)


# taken from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_value: float = 0.1,
        inplace: bool = False,
    ) -> None:
        super().__init__()

        self._inplace = inplace
        self._scale = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: Float[Tensor, "B ..."]) -> Float[Tensor, "B ..."]:
        return x.mul_(self._scale) if self._inplace else x * self._scale
