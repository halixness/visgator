##
##
##

from jaxtyping import Float
from torch import Tensor, nn

import deepsight.measures.functional as F
from deepsight.data.structs import BoundingBoxes
from deepsight.measures import Reduction


class GeneralizedBoxIoULoss(nn.Module):
    def __init__(self, reduction: Reduction = Reduction.MEAN) -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self, predictions: BoundingBoxes, targets: BoundingBoxes
    ) -> Float[Tensor, "..."]:
        return F.generalized_box_iou_loss(
            predictions=predictions,
            targets=targets,
            reduction=self.reduction,
        )

    def __call__(
        self, predictions: BoundingBoxes, targets: BoundingBoxes
    ) -> Float[Tensor, "..."]:
        return super().__call__(predictions, targets)  # type: ignore
