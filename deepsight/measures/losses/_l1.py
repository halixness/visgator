##
##
##

from jaxtyping import Float
from torch import Tensor, nn

import deepsight.measures.functional as F
from deepsight.data.structs import BoundingBoxes, BoundingBoxFormat
from deepsight.measures import Reduction


class BoxL1Loss(nn.Module):
    def __init__(
        self,
        format: BoundingBoxFormat | None = BoundingBoxFormat.CXCYWH,
        normalized: bool | None = True,
        reduction: Reduction = Reduction.MEAN,
    ) -> None:
        super().__init__()

        self.format = format
        self.normalized = normalized
        self.reduction = reduction

    def forward(
        self, predictions: BoundingBoxes, targets: BoundingBoxes
    ) -> Float[Tensor, "..."]:
        return F.box_l1_loss(
            predictions,
            targets,
            format=self.format,
            normalized=self.normalized,
            reduction=self.reduction,
        )

    def __call__(
        self, predictions: BoundingBoxes, targets: BoundingBoxes
    ) -> Float[Tensor, "..."]:
        return super().__call__(predictions, targets)  # type: ignore
