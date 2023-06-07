##
##
##

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from visgator.models import Criterion as _Criterion
from visgator.models import LossInfo
from visgator.utils.bbox import BBoxes


class Criterion(_Criterion[BBoxes]):
    def __init__(self) -> None:
        super().__init__()

    @property
    def losses(self) -> list[LossInfo]:
        return [LossInfo("l1_loss", 1.0)]

    def forward(self, output: BBoxes, target: BBoxes) -> dict[str, Float[Tensor, ""]]:
        output = output.to_xyxy().normalize()
        target = target.to_xyxy().normalize()
        loss = F.l1_loss(output.tensor, target.tensor)
        return {"l1_loss": loss}
