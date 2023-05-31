##
##
##

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torch_scatter import scatter_add

from visgator.utils.batch import Caption
from visgator.utils.bbox import BBoxes, BBoxFormat, ops
from visgator.utils.misc import Nested4DTensor

from ._config import DecoderConfig
from ._misc import CaptionEmbeddings, DetectionResults, Graph, NestedGraph
from ._position import GaussianHeatmap, Position2DEncodings, SpatialRelationEncodings


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()

        self._hidden_dim = config.hidden_dim
        self._same_entity_edge = nn.Parameter(torch.randn(1, config.hidden_dim))

        self._position_encodings = Position2DEncodings(config.hidden_dim)
        self._spatial_relation_encodings = SpatialRelationEncodings(config.hidden_dim)
        self._gaussian_heatmap = GaussianHeatmap()

        self._blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_layers)]
        )

    def forward(
        self,
        images: Nested4DTensor,
        captions: list[Caption],
        embeddings: list[CaptionEmbeddings],
        detections: list[DetectionResults],
    ) -> list[Graph]:
        graphs = [
            Graph.new(caption, embedding, detection, self._same_entity_edge)
            for caption, embedding, detection in zip(captions, embeddings, detections)
        ]

        graph = NestedGraph.from_graphs(graphs)
        max_nodes = max([nodes for nodes, _ in graph.sizes])  # N
        nodes_batch = len(graph) * max_nodes  # BN

        padded_boxes = detections[0].boxes.tensor.new_zeros(nodes_batch, 4)  # (BN, 4)
        for i, detection in enumerate(detections):
            boxes = detection.boxes.to_xyxy().normalize()
            start = i * max_nodes
            end = start + len(boxes)
            padded_boxes[start:end] = boxes.tensor

        boxes = BBoxes(
            padded_boxes,
            images_size=images.shape[2:],
            format=BBoxFormat.XYXY,
            normalized=True,
        )

        edge_index = graph.edge_index(False)  # (2 BE)
        boxes1 = boxes[edge_index[0]]  # (BE 4)
        boxes2 = boxes[edge_index[1]]  # (BE 4)
        union = ops.union_box_pairwise(boxes1.tensor, boxes2.tensor)
        union_boxes = BBoxes(
            union,
            images_size=images.shape[2:],
            format=BBoxFormat.XYXY,
            normalized=True,
        )

        heatmaps = self._gaussian_heatmap(boxes, images.shape[2:])  # (BN, HW)
        union_heatmaps = self._gaussian_heatmap(union_boxes, images.shape[2:])
        heatmaps1 = heatmaps[edge_index[0]]  # (BE HW)
        heatmaps2 = heatmaps[edge_index[1]]  # (BE HW)
        edge_heatmaps = torch.max(
            torch.max(heatmaps1, heatmaps2),
            union_heatmaps,
        )  # (BE HW)

        heatmaps = torch.log(heatmaps + 1e-8)  # (BN HW)
        union_heatmaps = torch.log(union_heatmaps + 1e-8)  # (BE HW)

        node_heatmaps = heatmaps.view(len(graph), max_nodes, -1)  # (B N HW)
        edge_heatmaps = edge_heatmaps.view(len(graph), -1)  # (B E HW)
        heatmaps = torch.cat((node_heatmaps, edge_heatmaps), dim=1)  # (B (N+E) HW)

        masks = images.mask.unsqueeze(1).expand(-1, heatmaps.shape[1], -1)
        masks = heatmaps.masked_fill_(masks, float("-inf"))  # (B (N+E) HW)
        edge_encodings = self._spatial_relation_encodings(boxes1, boxes2)  # (BE D)

        for block in self._blocks:
            graph = block(graph, edge_encodings, images, masks)

        return graph.to_graphs()

    def __call__(
        self,
        image: Nested4DTensor,
        captions: list[Caption],
        embeddings: list[CaptionEmbeddings],
        detections: list[DetectionResults],
    ) -> list[Graph]:
        return super().__call__(image, captions, embeddings, detections)  # type: ignore


class DecoderBlock(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()

        # attention with images
        self._ne_ln1 = nn.LayerNorm(config.hidden_dim)
        self._img_ln = nn.LayerNorm(config.hidden_dim)
        self._attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
        )
        self._ne_ls1 = LayerScale(config.hidden_dim)

        # attention with graph
        self._ne_ln2 = nn.LayerNorm(config.hidden_dim)
        self._gat = ModifiedGAT(config)
        self._ne_ls2 = LayerScale(config.hidden_dim)

        # feedforward
        self._ne_ln3 = nn.LayerNorm(config.hidden_dim)
        self._ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
        )
        self._ne_ls3 = LayerScale(config.hidden_dim)

    def forward(
        self,
        graph: NestedGraph,
        spatial_relation_encodings: Float[Tensor, "E D"],
        images: Float[Tensor, "B HW D"],
        mask: Float[Tensor, "BH (N+E) D"],
    ) -> NestedGraph:
        nodes, edges = graph.nodes(True), graph.edges(True)  # (B N D), (B E D)
        ne = torch.cat((nodes, edges), dim=1)  # (B (N+E) D)

        # image attention
        ln_ne = self._ne_ln1(ne)  # (B (N+E) D)
        ln_images = self._img_ln(images)  # (B HW D)
        attn_output, _ = self._attn(
            ln_ne,
            ln_images,
            ln_images,
            attn_mask=mask,
            need_weights=False,
        )  # (B (N+E) D)
        ne = ne + self._ne_ls1(attn_output)  # (B (N+E) D)

        # graph attention
        ln_ne = self._ne_ln2(ne)  # (B (N+E) D)
        nodes, edges = ln_ne.split(
            [nodes.shape[1], edges.shape[1]], dim=1
        )  # (B N D), (B E D)
        graph = graph.new_like(nodes, edges)
        graph = self._gat(graph, spatial_relation_encodings)
        nodes, edges = graph.nodes(True), graph.edges(True)  # (B N D), (B E D)
        gat_ne = torch.cat((nodes, edges), dim=1)  # (B (N+E) D)
        ne = ne + self._ne_ls2(gat_ne)  # (B (N+E) D)

        # feedforward
        ln_ne = self._ne_ln3(ne)  # (B (N+E) D)
        ffn_output = self._ffn(ln_ne)  # (B (N+E) D)
        ne = ne + self._ne_ls3(ffn_output)  # (B (N+E) D)

        nodes, edges = ne.split(
            [nodes.shape[1], edges.shape[1]], dim=1
        )  # (B N D), (B E D)

        return graph.new_like(nodes, edges)


# Implementation based on GATv2Conv of Pytorch Geometric
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gatv2_conv.py
class ModifiedGAT(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()

        self._num_heads = config.num_heads

        self._src_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self._dst_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self._edge_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self._attn_proj = nn.Linear(config.hidden_dim, 1)

        self._edge_out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self._node_out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        self._dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        graph: NestedGraph,
        spatial_edge_encoding: Float[Tensor, "BE D"],
    ) -> NestedGraph:
        nodes = graph.nodes(False)  # (BN D)
        edges = graph.edges(False)  # (BE D)
        edge_index = graph.edge_index(False)  # (2 BE)

        BN, _ = nodes.shape
        H = self._num_heads

        x_src = self._src_proj(nodes).view(BN, H, -1)  # (BN H D/H)
        x_dst = self._dst_proj(nodes).view(BN, H, -1)  # (BN H D/H)
        edges = self._edge_proj(edges).view(BN, H, -1)  # (BE H D/H)

        x_src = x_src[edge_index[0]]  # (BE H D/H)
        x_dst = x_dst[edge_index[1]]  # (BE H D/H)

        tmp = x_src + x_dst + edges + spatial_edge_encoding  # (BE H D/H)
        tmp = F.gelu(tmp)  # (BE H D/H)
        presoftmax_alpha = self._attn_proj(tmp).squeeze(-1)  # (BE H)

        exp_alpha = torch.exp(presoftmax_alpha)  # (BE H)
        alpha_sum = scatter_add(exp_alpha, edge_index[0], dim=0)  # (BN H)
        alpha = exp_alpha / alpha_sum[edge_index[0]]  # (BE H)
        alpha = self._dropout(alpha)  # (BE H)

        values = self._node_out_proj(nodes).view(BN, H, -1)  # (BN H D/H)
        values = values[edge_index[1]]  # (BE H D/H)
        values = alpha.unsqueeze(-1) * values  # (BE H D/H)

        new_nodes = scatter_add(values, edge_index[0], dim=0)  # (BN H D/H)
        new_nodes = new_nodes.view(BN, -1)  # (BN D)
        new_edges = self._edge_out_proj(tmp).view(BN, -1)  # (BE D)

        return graph.new_like(new_nodes, new_edges)


# taken from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_value: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()

        self._inplace = inplace
        self._scale = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: Float[Tensor, "B ..."]) -> Float[Tensor, "B ..."]:
        return x.mul_(self._scale) if self._inplace else x * self._scale
