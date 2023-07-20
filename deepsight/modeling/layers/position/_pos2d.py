##
##
##

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn


# code based on https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/position_embedding.py
class Sinusoidal2DPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for 2D feature maps.

    For each position in the input feature map, this module computes an embedding
    vector for its x and y coordinates using sinusoidal functions as described in
    `"Attention Is All You Need" [1_]. The embedding vectors are then concatenated
    to form the final position embedding vector.

    .. note::
        Since such position embeddings are intended to be matched with the ones
        computed for the bounding boxes, the first half of the embedding vector
        corresponds to the x coordinate and the second half to the y coordinate.
        This is the opposite of other implementations such as the one in
        `detrex <https://github.com/IDEA-Research/detrex>`_.


    Attributes
    ----------
    dim : int
        The final embedding dimension. This is equal to twice the feature dimension
        used for each coordinate.
    temperature : float
        The temperature of the sinusoidal function. Defaults to `20`.

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N.,
        Kaiser, Ł. and Polosukhin, I., 2017. Attention is all you need.
        Advances in neural information processing systems, 30.
    """

    def __init__(
        self,
        dim: int,
        temperature: int = 20,
        scale: float = 2 * torch.pi,
        offset: float = 0.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}.")

        self.dim = dim
        self.temperature = temperature
        self.scale = scale
        self.offset = offset
        self.normalize = normalize

    def forward(self, mask: Bool[Tensor, "B H W"]) -> Float[Tensor, "B C H W"]:
        dim = self.dim // 2
        temperature = self.temperature

        not_mask = ~mask

        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)  # (B, H, W)
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)  # (B, H, W)

        if self.normalize:
            eps = torch.finfo(y_embed.dtype).eps
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(dim, dtype=torch.float32, device=mask.device)  # (C / 2)
        dim_t = temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / dim
        )  # (C / 2)

        pos_x = x_embed[:, :, :, None] / dim_t  # (B, H, W, C / 2)
        pos_y = y_embed[:, :, :, None] / dim_t  # (B, H, W, C / 2)

        B, H, W = mask.size()
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4)
        pos_x = pos_x.view(B, H, W, -1)

        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4)
        pos_y = pos_y.view(B, H, W, -1)

        pos = torch.cat((pos_x, pos_y), dim=3).permute(0, 3, 1, 2)  # (B, C, H, W)

        return pos

    def __call__(self, mask: Bool[Tensor, "B H W"]) -> Float[Tensor, "B C H W"]:
        """Computes the position embeddings for a batch of feature maps.

        Parameters
        ----------
        mask : Bool[Tensor, "B H W"]
            A boolean tensor of shape (B, H, W) where B is the batch size, H is the
            height of the feature map and W is the width of the feature map. The
            value True indicates that the corresponding position in the feature map
            is masked out.

        Returns
        -------
        Float[Tensor, "B C H W"]
            The position embeddings for the input feature maps.
        """
        return super().__call__(mask)  # type: ignore
