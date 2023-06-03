##
##
##

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from visgator.models import Criterion as _Criterion
from visgator.utils.bbox import BBoxes, ops

from .._criterion import LossInfo
from ._config import CriterionConfig


class Criterion(_Criterion[BBoxes]):
    def __init__(self, config: CriterionConfig) -> None:
        super().__init__()

        self._l1_weight = config.l1_weight
        self._giou_weight = config.giou_weight

    @property
    def losses(self) -> list[LossInfo]:
        return [
            LossInfo("L1", self._l1_weight),
            LossInfo("GIoU", self._giou_weight),
        ]

    def forward(self, output: BBoxes, target: BBoxes) -> dict[str, Float[Tensor, ""]]:
        output = output.to_xyxy().normalize()
        target = target.to_xyxy().normalize()

        l1_loss = F.l1_loss(output.tensor, target.tensor)
        gious_loss = -ops.generalized_box_iou_pairwise(
            output.tensor, target.tensor
        ).mean()

        return {
            "L1": l1_loss,
            "GIoU": gious_loss,
        }
