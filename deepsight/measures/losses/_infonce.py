##
##
##

from jaxtyping import Float
from torch import Tensor, nn

import deepsight.measures.functional as F
from deepsight.measures import Reduction


class InfoNCELoss(nn.Module):
    def __init__(
        self, temperature: float = 0.1, reduction: Reduction = Reduction.MEAN
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        queries: Float[Tensor, "N D"],
        positive_keys: Float[Tensor, "N D"],
        negative_keys: Float[Tensor, "M D"],
    ) -> Float[Tensor, "..."]:
        return F.infonce_loss(
            queries,
            positive_keys,
            negative_keys,
            temperature=self.temperature,
            reduction=self.reduction,
        )

    def __call__(
        self,
        queries: Float[Tensor, "N D"],
        positive_keys: Float[Tensor, "N D"],
        negative_keys: Float[Tensor, "M D"],
    ) -> Float[Tensor, "..."]:
        return super().__call__(queries, positive_keys, negative_keys)  # type: ignore
