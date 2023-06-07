##
##
##

"""Utilities for bounding box operations."""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchvision.ops.boxes import box_area


# taken from https://github.com/IDEA-Research/DINO/blob/main/util/box_ops.py
def box_iou_pairwise(
    boxes1: Float[Tensor, "N 4"],
    boxes2: Float[Tensor, "N 4"],
) -> tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:  # noqa: F821
    """Computes the iou between two boxes.

    Parameters
    ----------
    boxes1 : Float[Tensor, "N 4"]
        The first boxes in [x1, y1, x2, y2] format.
    boxes2 : Float[Tensor, "N 4"]
        The second boxes in [x1, y1, x2, y2] format.

    Returns
    -------
    tuple[Float[Tensor, "N"], Float[Tensor, "N"]]
        The iou and union area of the two boxes.
    """

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


# taken from https://github.com/IDEA-Research/DINO/blob/main/util/box_ops.py
def generalized_box_iou_pairwise(
    boxes1: Float[Tensor, "N 4"],
    boxes2: Float[Tensor, "N 4"],
) -> Float[Tensor, "N"]:  # noqa: F821
    """Computes the generalized box iou between two boxes as described in
    https://giou.stanford.edu/.

    Parameters
    ----------
    boxes1 : Float[Tensor, "N 4"]
        The first boxes in [x1, y1, x2, y2] format.
    boxes2 : Float[Tensor, "N 4"]
        The second boxes in [x1, y1, x2, y2] format.

    Returns
    -------
    Float[Tensor, "N"]
        The generalized box iou between the two boxes.
    """

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


def union_box_pairwise(
    boxes1: Float[Tensor, "N 4"],
    boxes2: Float[Tensor, "N 4"],
) -> Float[Tensor, "N 4"]:
    """Computes the union of two boxes.

    Parameters
    ----------
    boxes1 : Float[Tensor, "N 4"]
        The first boxes in [x1, y1, x2, y2] format.
    boxes2 : Float[Tensor, "N 4"]
        The second boxes in [x1, y1, x2, y2] format.

    Returns
    -------
    Float[Tensor, "N 4"]
        The union of the two boxes in [x1, y1, x2, y2] format.
    """

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])  # (N, 2)
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])  # (N, 2)
    return torch.cat([lt, rb], dim=-1)


def from_xyxy_to_xywh(boxes: Float[Tensor, "N 4"]) -> Float[Tensor, "N 4"]:
    """Converts [x1, y1, x2, y2] box format to [x, y, w, h] format.

    Parameters
    ----------
    boxes : Float[Tensor, "N 4"]
        The boxes in [x1, y1, x2, y2] format.

    Returns
    -------
    Float[Tensor, "N 4"]
        The boxes in [x, y, w, h] format.
    """
    top_left = boxes[:, :2]
    size = boxes[:, 2:] - boxes[:, :2]
    return torch.cat([top_left, size], dim=-1)


def from_xywh_to_xyxy(boxes: Float[Tensor, "N 4"]) -> Float[Tensor, "N 4"]:
    """Converts [x, y, w, h] box format to [x1, y1, x2, y2] format.

    Parameters
    ----------
    boxes : Float[Tensor, "N 4"]
        The boxes in [x, y, w, h] format.

    Returns
    -------
    Float[Tensor, "N 4"]
        The boxes in [x1, y1, x2, y2] format.
    """
    top_left = boxes[:, :2]
    bottom_right = boxes[:, :2] + boxes[:, 2:]
    return torch.cat([top_left, bottom_right], dim=-1)


def from_xywh_to_cxcywh(boxes: Float[Tensor, "N 4"]) -> Float[Tensor, "N 4"]:
    """Converts [x, y, w, h] box format to [cx, cy, w, h] format.

    Parameters
    ----------
    boxes : Float[Tensor, "N 4"]
        The boxes in [x, y, w, h] format.

    Returns
    -------
    Float[Tensor, "N 4"]
        The boxes in [cx, cy, w, h] format.
    """
    center = boxes[:, :2] + (boxes[:, 2:] / 2)
    size = boxes[:, 2:]
    return torch.cat([center, size], dim=-1)


def from_cxcywh_to_xywh(boxes: Float[Tensor, "N 4"]) -> Float[Tensor, "N 4"]:
    """Converts [cx, cy, w, h] box format to [x, y, w, h] format.

    Parameters
    ----------
    boxes : Float[Tensor, "N 4"]
        The boxes in [cx, cy, w, h] format.

    Returns
    -------
    Float[Tensor, "N 4"]
        The boxes in [x, y, w, h] format.
    """
    top_left = boxes[:, :2] - (boxes[:, 2:] / 2)
    size = boxes[:, 2:]
    return torch.cat([top_left, size], dim=-1)


def from_xyxy_to_cxcywh(boxes: Float[Tensor, "N 4"]) -> Float[Tensor, "N 4"]:
    """Converts [x1, y1, x2, y2] box format to [cx, cy, w, h] format.

    Parameters
    ----------
    boxes : Float[Tensor, "N 4"]
        The boxes in [x1, y1, x2, y2] format.

    Returns
    -------
    Float[Tensor, "N 4"]
        The boxes in [cx, cy, w, h] format.
    """
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    size = boxes[:, 2:] - boxes[:, :2]
    return torch.cat([center, size], dim=-1)


def from_cxcywh_to_xyxy(boxes: Float[Tensor, "N 4"]) -> Float[Tensor, "N 4"]:
    """Converts [cx, cy, w, h] box format to [x1, y1, x2, y2] format.

    Parameters
    ----------
    boxes : Float[Tensor, "N 4"]
        The boxes in [cx, cy, w, h] format.

    Returns
    -------
    Float[Tensor, "N 4"]
        The boxes in [x1, y1, x2, y2] format.
    """
    top_left = boxes[:, :2] - (boxes[:, 2:] / 2)
    bottom_right = boxes[:, :2] + (boxes[:, 2:] / 2)
    return torch.cat([top_left, bottom_right], dim=-1)


def normalize(
    boxes: Float[Tensor, "N 4"], image_sizes: Int[Tensor, "N 2"]
) -> Float[Tensor, "N 4"]:
    """Normalizes boxes to [0, 1] range.

    Parameters
    ----------
    boxes : Float[Tensor, "N 4"]
        Unnormalized boxes in any format.
    image_sizes : Int[Tensor, "N 2"]
        Image sizes in [H, W] format.

    Returns
    -------
    Float[Tensor, "N 4"]
        Normalized boxes in [0, 1] range with the same format as input.
    """
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2] / image_sizes[:, 1].unsqueeze(-1)
    boxes[:, 1::2] = boxes[:, 1::2] / image_sizes[:, 0].unsqueeze(-1)
    return boxes


def denormalize(
    boxes: Float[Tensor, "N 4"], image_sizes: Int[Tensor, "N 2"]
) -> Float[Tensor, "N 4"]:
    """Denormalizes boxes from [0, 1] range to image range.

    Parameters
    ----------
    boxes : Float[Tensor, "N 4"]
        Normalized boxes in [0, 1] range.
    image_sizes : Int[Tensor, "N 2"]
        Image sizes in [H, W] format.

    Returns
    -------
    Float[Tensor, "N 4"]
        Unnormalized boxes with the same format as input.
    """
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2] * image_sizes[:, 1].unsqueeze(-1)
    boxes[:, 1::2] = boxes[:, 1::2] * image_sizes[:, 0].unsqueeze(-1)
    return boxes
