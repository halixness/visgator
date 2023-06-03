##
##
##

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from visgator.utils.bbox import BBoxes, BBoxFormat

from ._config import HeadConfig
from ._misc import NestedGraph


class RegressionHead(nn.Module):
    def __init__(self, config: HeadConfig) -> None:
        super().__init__()

        self._dim = config.hidden_dim
        self._token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))

        self._layers = nn.ModuleList(
            [ResidualAttentionLayer(config) for _ in range(config.num_layers)]
        )

        self._regression_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 4),
            nn.Sigmoid(),
        )

    def _positional_encoding(self, mask: Bool[Tensor, "B L"]) -> Float[Tensor, "B L D"]:
        not_mask = ~mask  # (B, L)
        embed = not_mask.cumsum(dim=1, dtype=torch.float32)  # (B, L)

        dim_t = torch.arange(self._dim, dtype=torch.float32, device=mask.device)
        dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self._dim)

        pos = embed[:, :, None] / dim_t  # (B, L, D)
        pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=3)
        pos = pos.view(pos.shape[0], pos.shape[1], -1)  # (B, L, D)

        return pos

    def forward(self, graph: NestedGraph, images_size: list[tuple[int, int]]) -> BBoxes:
        B = len(graph)
        nodes = graph.nodes(True)  # (B, N, D)
        edges = graph.edges(True)  # (B, E, D)
        tokens = torch.cat([nodes, edges], dim=1)  # (B, N+E, D)
        N, E = nodes.shape[1], edges.shape[1]

        mask = nodes.new_ones((B, 1 + N + E), dtype=torch.bool)  # (B, 1+N+E)
        for idx, (num_nodes, num_edges) in enumerate(graph.sizes):
            mask[idx, 0] = False
            mask[idx, 1 : 1 + num_nodes] = False
            mask[idx, 1 + N : 1 + N + num_edges] = False

        tokens = tokens + self._positional_encoding(mask[:, 1:])  # (B, N+E, D)
        x = torch.cat([self._token.expand(B, -1, -1), tokens], dim=1)  # (B, 1+N+E, D)

        for layer in self._layers:
            x = layer(x, mask)

        token = self._regression_head(x[:, 0])  # (B, 4)
        boxes = BBoxes(token, images_size, BBoxFormat.CXCYWH, True)  # (B, 4)

        return boxes

    def __call__(
        self, graph: NestedGraph, images_size: list[tuple[int, int]]
    ) -> BBoxes:
        return super().__call__(graph, images_size)  # type: ignore


class ResidualAttentionLayer(nn.Module):
    def __init__(self, config: HeadConfig) -> None:
        super().__init__()

        self._norm1 = nn.LayerNorm(config.hidden_dim)
        self._attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self._dropout1 = nn.Dropout(config.dropout)

        self._norm2 = nn.LayerNorm(config.hidden_dim)
        self._ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
        )
        self._dropout2 = nn.Dropout(config.dropout)

    def forward(
        self, x: Float[Tensor, "B L D"], mask: Bool[Tensor, "B L"]
    ) -> Float[Tensor, "B L D"]:
        x1 = self._norm1(x)
        x1 = self._attn(x1, x1, x1, key_padding_mask=mask, need_weights=False)[0]
        x1 = self._dropout1(x1)
        x1 = x + x1

        x2 = self._norm2(x1)
        x2 = self._ffn(x2)
        x2 = self._dropout2(x2)
        x2 = x1 + x2

        return x2  # type: ignore
