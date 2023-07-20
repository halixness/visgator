##
##
##

import torch
import torchmetrics as tm
from jaxtyping import Float
from torch import Tensor

from deepsight.data.structs import BoundingBoxes


class GeneralizedBoxIoU(tm.Metric):
    """Generalized Intersection over Union (GIoU) metric for bounding boxes."""

    higher_is_better = True
    is_differentiable = True

    def __init__(self) -> None:
        super().__init__()

        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.value: torch.Tensor  # just to make mypy happy
        self.total: torch.Tensor  # just to make mypy happy

    def update(self, predictions: BoundingBoxes, targets: BoundingBoxes) -> None:
        self.value += predictions.generalized_iou(targets).sum()
        self.total += predictions.numel()

    def compute(self) -> Float[Tensor, ""]:
        return self.value / self.total
