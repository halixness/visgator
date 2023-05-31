##
##
##

from jaxtyping import Float
from torch import Tensor

from visgator.utils.bbox import BBoxes

from .._criterion import Criterion as _Criterion
from .._criterion import LossInfo


class Criterion(_Criterion[BBoxes]):
    @property
    def losses(self) -> list[LossInfo]:
        return [LossInfo("L1", 1.0), LossInfo("GIoU", 1.0)]

    def forward(self, output: BBoxes, target: BBoxes) -> dict[str, Float[Tensor, ""]]:
        raise NotImplementedError
