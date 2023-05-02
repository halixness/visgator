## borrowed from https://github.com/IDEA-Research/DINO/blob/main/util/box_ops.py
##
##

"""Utilities for bounding box operations."""

import torch
from jaxtyping import Float
from torch import Tensor
from torchvision.ops.boxes import box_area


def box_iou_pairwise(
    boxes1: Float[Tensor, "N 4"], boxes2: Float[Tensor, "N 4"]
) -> tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:  # noqa: F821
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou_pairwise(
    boxes1: Float[Tensor, "N 4"], boxes2: Float[Tensor, "N 4"]
) -> Float[Tensor, "N"]:  # noqa: F821
    """Generalized IoU from https://giou.stanford.edu/"""
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape == boxes2.shape
    iou, union = box_iou_pairwise(boxes1, boxes2)  # [N,4]

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,2]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area
