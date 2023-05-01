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
) -> Float[Tensor, "N"]:  # noqa: F821
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou  # type: ignore
