##
##
##

import torch
from jaxtyping import Float
from torch import Tensor, nn


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

    def __call__(self, x: Float[Tensor, "B ..."]) -> Float[Tensor, "B ..."]:
        return super().__call__(x)  # type: ignore
