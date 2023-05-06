##
##
##

import torch
from jaxtyping import Float
from torch import Tensor


def xywh_to_xyxy(xywh: Float[Tensor, "4"]) -> Float[Tensor, "4"]:
    """Convert [x, y, w, h] box format to [x1, y1, x2, y2] format."""
    x, y, w, h = xywh.unbind(-1)
    xyxy = [x, y, x + w, y + h]
    return torch.stack(xyxy, dim=-1)


def cxcywh_to_xyxy(cxcywh: Float[Tensor, "4"]) -> Float[Tensor, "4"]:
    """Convert [cx, cy, w, h] box format to [x1, y1, x2, y2] format."""
    cx, cy, w, h = cxcywh.unbind(-1)
    xyxy = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    return torch.stack(xyxy, dim=-1)


def xyxyn_to_xyxy(
    xyxyn: Float[Tensor, "4"], image_size: tuple[int, int]
) -> Float[Tensor, "4"]:
    """Convert [x1, y1, x2, y2] normalized box format to [x1, y1, x2, y2] format."""
    h, w = image_size
    return xyxyn * xyxyn.new_tensor([w, h, w, h])


def xywhn_to_xyxy(
    xywhn: Float[Tensor, "4"], image_size: tuple[int, int]
) -> Float[Tensor, "4"]:
    """Convert [x, y, w , h] normalized box format to [x1, y1, x2, y2] format."""
    h, w = image_size
    xywh = xywhn * xywhn.new_tensor([w, h, w, h])
    return xywh_to_xyxy(xywh)


def cxcywhn_to_xyxy(
    cxcywhn: Float[Tensor, "4"], image_size: tuple[int, int]
) -> Float[Tensor, "4"]:
    """Convert [cx, cy, w , h] normalized box format to [x1, y1, x2, y2] format."""
    h, w = image_size
    cxcywh = cxcywhn * cxcywhn.new_tensor([w, h, w, h])
    return cxcywh_to_xyxy(cxcywh)
