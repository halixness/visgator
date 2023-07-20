##
##
##

import torch
from jaxtyping import Float
from torch import Tensor, nn

from deepsight.data.structs import BoundingBoxes


class SinusoidalPairwiseBoxEmbeddings(nn.Module):
    """Sinusoidal embeddings for pairs of bounding boxes."""

    def __init__(self, dim: int, temperature: int = 20) -> None:
        super().__init__()

        if dim % 4 != 0:
            raise ValueError(f"dim must be divisible by 4, got {dim}.")

        self.dim = dim
        self.temperature = temperature

    def forward(
        self,
        first: BoundingBoxes,
        second: BoundingBoxes,
    ) -> Float[Tensor, "... D"]:
        first = first.to_cxcywh().normalize()
        second = second.to_cxcywh().normalize()

        distance = first.tensor[..., :2] - second.tensor[..., :2]  # (..., 2)
        iou = first.iou(second).unsqueeze(-1)  # (..., 1)
        union = first.union(second).area().unsqueeze(-1)  # (..., 1)

        coords = torch.cat((distance, iou, union), dim=-1)  # (..., 4)
        coords = coords.unsqueeze(-1)  # (..., 4, 1)

        dim = self.dim // 4  # D
        temperature = self.temperature

        dim_t = torch.arange(dim, dtype=torch.float32, device=coords.device)  # (D,)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / dim)

        pos = coords / dim_t
        pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1)
        pos = pos.flatten(start_dim=-3)  # (..., D)

        return pos

    def __call__(
        self,
        first: BoundingBoxes,
        second: BoundingBoxes,
    ) -> Float[Tensor, "... D"]:
        return super().__call__(first, second)  # type: ignore
