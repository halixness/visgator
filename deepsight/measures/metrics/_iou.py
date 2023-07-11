##
##
##

import torch
import torchmetrics as tm
from jaxtyping import Float
from torch import Tensor

from deepsight.data.structs import BoundingBoxes


class BoxIoU(tm.Metric):
    """Intersection over Union (IoU) metric for bounding boxes."""

    higher_is_better = True
    is_differentiable = True

    def __init__(self) -> None:
        super().__init__()

        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.value: torch.Tensor  # just to make mypy happy
        self.total: torch.Tensor  # just to make mypy happy

    def update(self, predictions: BoundingBoxes, targets: BoundingBoxes) -> None:
        self.value += predictions.iou(targets).sum()
        self.total += predictions.numel()

    def compute(self) -> Float[Tensor, ""]:
        return self.value / self.total


class BoxIoUAccuracy(tm.Metric):
    """Intersection over Union (IoU) accuracy metric for bounding boxes.

    This metric computes the accuracy of the IoU metric, i.e. the percentage of
    bounding boxes that have an IoU greater than or equal to a given threshold.
    """

    higher_is_better = True
    is_differentiable = True

    def __init__(self, threshold: float) -> None:
        super().__init__()

        self.register_buffer("threshold", torch.tensor(threshold))
        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.threshold: torch.Tensor  # just to make mypy happy
        self.value: torch.Tensor  # just to make mypy happy
        self.total: torch.Tensor  # just to make mypy happy

    def update(self, predictions: BoundingBoxes, targets: BoundingBoxes) -> None:
        iou = predictions.iou(targets)

        self.value += (iou >= self.threshold).sum()
        self.total += predictions.numel()

    def compute(self) -> Float[Tensor, ""]:
        return self.value / self.total
