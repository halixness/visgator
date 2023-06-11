##
##
##

import torch
from jaxtyping import Float
from torch import Tensor, nn

from visgator.utils.bbox import BBoxes
from visgator.utils.torch import Nested4DTensor

from ._config import DecoderConfig
from ._misc import NestedGraph
from ._position import GaussianHeatmaps


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()

        self._num_heads = config.num_heads
        self._hidden_dim = config.hidden_dim
        self._same_entity_edge = nn.Parameter(torch.randn(1, config.hidden_dim))

        # self._patch_encondings = PatchSpatialEncodings(config.hidden_dim)
        self._gaussian_heatmaps = GaussianHeatmaps()

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

        # Problem found: edge_index can point to samples out of the boxes tensor. Probs because we have discarded some samples
        # print(f"\n----nested graph edge index: {graph._edge_index}")
        # print(f"boxes shape : {boxes._boxes.shape}\n----")

        # (entity1, entity2), edges
        edge_index = graph.edge_index(False)  # (2 BE)

        # Filter to avoid out of bounds
        # Filtering leads to tensors of odd shapes which causes a problem in the union
        edge_mask = edge_index < boxes._boxes.shape[0]
        new_edge_index = None
        for k, ind in enumerate(edge_index):
            if new_edge_index == None: new_edge_index = ind[edge_mask[k]].unsqueeze(0)
            else: new_edge_index = torch.concat((new_edge_index, ind[edge_mask[k]].unsqueeze(0)))
        edge_index = new_edge_index

        # Select bboxes that have a connection
        boxes1 = boxes[edge_index[0]]  # (BE 4)
        boxes2 = boxes[edge_index[1]]  # (BE 4)
    
        union_boxes = boxes1.union(boxes2)  # (BE 4)

        print(f"\nboxes1: {boxes1._boxes.shape}")
        print(f"boxes2: {boxes2._boxes.shape}")
        print(f"union: {union_boxes._boxes.shape}")
        print(f"boxes: {boxes._boxes.shape}")

        heatmaps = self._gaussian_heatmaps(boxes, (H, W))  # (BN HW)
        union_heatmaps = self._gaussian_heatmaps(union_boxes, (H, W)) # (BE, HW)
        heatmaps1 = heatmaps[edge_index[0]]  # (BE HW)
        heatmaps2 = heatmaps[edge_index[1]]  # (BE HW)
        
        print(f"heatmaps1: {heatmaps1.shape}")
        print(f"heatmaps2: {heatmaps2.shape}")
        print(f"union_heatmaps: {union_heatmaps.shape}")
        print(f"edge_index : {edge_index}")

        edge_heatmaps = torch.maximum(
            torch.maximum(heatmaps1, heatmaps2),
            union_heatmaps,
        )  # (BE HW)

        heatmaps = torch.log(heatmaps + 1e-8)  # (BN HW)
        edge_heatmaps = torch.log(edge_heatmaps + 1e-8)  # (BE HW)
        
        print(f"node_heatmaps: {heatmaps.shape}")
        print(f"edge_heatmaps: {edge_heatmaps.shape}")

        node_heatmaps = heatmaps.view(len(graph), -1, H * W)  # (B N HW)
        edge_heatmaps = edge_heatmaps.view(len(graph), -1, H * W)  # (B E HW)
        heatmaps = torch.cat((node_heatmaps, edge_heatmaps), dim=1)  # (B (N+E) HW)

        flattened_images = images.flatten()  # (B HW D)
        masks = flattened_images.mask.unsqueeze(1).expand(-1, heatmaps.shape[1], -1)
        masks = heatmaps.masked_fill_(masks, -torch.inf)  # (B (N+E) HW)
        masks = masks.repeat(self._num_heads, 1, 1)  # (Bh (N+E) HW)

        # image_encodings = self._patch_encondings(images.mask)

        nodes = graph.nodes(True)  # (B N D)
        edges = graph.edges(True)  # (B E D)
        x = torch.cat((nodes, edges), dim=1)  # (B (N+E) D)

        for block in self._layers:
            x = block(
                x,
                flattened_images.tensor,
                masks,
            )

        nodes = x[:, : nodes.shape[1]]  # (B N D)
        edges = x[:, nodes.shape[1] :]  # (B E D)

        return graph.new_like(nodes, edges)

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

        # feedforward
        self._norm2 = nn.LayerNorm(config.hidden_dim)
        self._ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
        )
        self._layerscale2 = LayerScale(config.hidden_dim, config.epsilon_layer_scale)

    def forward(
        self,
        x: Float[Tensor, "B (N+E) D"],
        images: Float[Tensor, "B HW D"],
        mask: Float[Tensor, "Bh (N+E) HW"],
    ) -> Float[Tensor, "B (N+E) D"]:
        # image attention
        x1 = self._norm1(x)  # (B (N+E) D)
        x1, _ = self._attn(
            x1,
            images,
            images,
            attn_mask=mask,
            need_weights=False,
        )  # (B (N+E) D)
        x1 = x + self._layerscale1(x1)  # (B (N+E) D)

        # feedforward
        x2 = self._norm2(x1)  # (B (N+E) D)
        x2 = self._ffn(x2)  # (B (N+E) D)
        x2 = x1 + self._layerscale2(x2)  # (B (N+E) D)

        return x2  # type: ignore


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
