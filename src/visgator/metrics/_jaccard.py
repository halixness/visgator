##
##
##

import torch
from jaxtyping import Float
from torch import Tensor
from torchmetrics import Metric
from visgator.utils.bbox import ops


class JaccardIndex(Metric):
    is_differentiable = True
    higher_is_better = True

    def __init__(self) -> None:
        super().__init__()
        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Float[Tensor, "N 4"], target: Float[Tensor, "N 4"]) -> None:
        iou = ops.box_iou_pairwise(pred, target)

        self.value += iou.sum()
        self.total += pred.shape[0]  # type: ignore

    def compute(self) -> torch.Tensor:
        return self.value.float() / self.total  # type: ignore
