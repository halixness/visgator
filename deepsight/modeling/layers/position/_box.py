##
##
##

import torch
from jaxtyping import Float
from torch import Tensor, nn

from deepsight.data.structs import BoundingBoxes


class SinusoidalBoxEmbeddings(nn.Module):
    """Sinusoidal box embeddings.

    This module computes sinusoidal embeddings for a set of bounding boxes. The
    embeddings are computed by applying sinusoidal functions (as described in [1]_)
    to the coordinates of the boxes and concatenating the results.

    .. note::
        Since such position embeddings are intended to be matched with the ones
        computed for the feature maps, the coordinates embeddings are concatenated
        in the following order: cx, cy, (w, h).

    Attributes
    ----------
    dim : int
        The final embedding dimension. This is equal to the feature dimension used
        for each coordinate times the number of coordinates used (2 or 4). If
        `include_wh` is `True`, the dimension must be divisible by 4, otherwise it
        must be even.
    temperature : float
        The temperature of the sinusoidal function. Defaults to `20`.
    include_wh : bool
        Whether to include the width and height of the boxes in the embeddings.
        Defaults to `False`.
    """

    def __init__(
        self,
        dim: int,
        temperature: int = 20,
        scale: float = 2 * torch.pi,
        include_wh: bool = False,
    ) -> None:
        super().__init__()

        if include_wh:
            if dim % 4 != 0:
                raise ValueError(f"dim must be divisible by 4, got {dim}.")
        else:
            if dim % 2 != 0:
                raise ValueError(f"dim must be even, got {dim}.")

        self.dim = dim
        self.temperature = temperature
        self.scale = scale
        self.include_wh = include_wh

    def forward(self, boxes: BoundingBoxes) -> Float[Tensor, "... D"]:
        boxes = boxes.to_cxcywh().normalize()

        dim = self.dim // 4 if self.include_wh else self.dim // 2  # D
        temperature = self.temperature

        dim_t = torch.arange(dim, dtype=torch.float32, device=boxes.device)  # (D,)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / dim)

        if self.include_wh:
            coords = boxes.tensor  # (..., 4)
        else:
            coords = boxes.tensor[..., :2]  # (..., 2)

        pos = coords.unsqueeze(-1) * self.scale / dim_t
        pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1)
        pos = pos.flatten(start_dim=-3)  # (..., D)

        return pos

    def __call__(self, boxes: BoundingBoxes) -> Float[Tensor, "... D"]:
        """Computes the embeddings for the given boxes.

        Parameters
        ----------
        boxes : BoundingBoxes
            The bounding boxes to compute the embeddings for. The bounding boxes
            tensor can have any number of leading dimension.

        Returns
        -------
        Float[Tensor, "... D"]
            The computed embeddings. The number of leading dimensions of the returned
            tensor is equal to the number of leading dimensions of the bounding boxes
            tensor. The last dimension is equal to the embedding dimension.
        """

        return super().__call__(boxes)  # type: ignore
