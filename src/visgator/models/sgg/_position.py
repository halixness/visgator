##
##
##

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from visgator.utils.bbox import BBoxes, ops


# Code taken from https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/position_embedding.py
class Position2DEncodings(nn.Module):
    """Positional sinusoidal encodings for 2D images."""

    def __init__(self, hidden_dim: int, temperature: int = 10000) -> None:
        super().__init__()

        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be divisible by 2.")
        self._dim = hidden_dim // 2

        self._temperature = temperature

    def forward(self, mask: Bool[Tensor, "B H W"]) -> Float[Tensor, "B C H W"]:
        not_mask = ~mask  # (B, H, W)
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)  # (B, H, W)
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)  # (B, H, W)

        dim_t = torch.arange(
            self._dim, dtype=torch.float32, device=mask.device
        )  # (C / 2)
        dim_t = self._temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self._dim
        )  # (C / 2)

        pos_x = x_embed[:, :, :, None] / dim_t  # (B, H, W, C / 2)
        pos_y = y_embed[:, :, :, None] / dim_t  # (B, H, W, C / 2)

        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).view(B, H, W, -1)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (B, C, H, W)

        return pos

    def __call__(self, mask: Bool[Tensor, "B H W"]) -> Float[Tensor, "B C H W"]:
        return super.__call__(mask)  # type: ignore


class GaussianHeatmap(nn.Module):
    """Gaussian heatmaps for 2D images."""

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()

        self._beta = beta

    def forward(self, boxes: BBoxes, size: tuple[int, int]) -> Float[Tensor, "B HW"]:
        boxes = boxes.to_xyxy().normalize()  # (B, 4)

        mean = boxes.tensor[:, :2]  # (B, 2)
        std = boxes.tensor[:, 2:]  # (B, 2)

        height, width = size
        coords = torch.cartesian_prod(
            torch.arange(height, device=boxes.device),
            torch.arange(width, device=boxes.device),
        )

        coords = coords / torch.tensor([height, width], device=boxes.device)

        coords = coords[None].expand(len(boxes), -1, -1)  # (B, HW, 2)

        x = (coords[:, :, 1] - mean[:, 0].unsqueeze(-1)) ** 2
        x = x / (self._beta * (std[:, 0].unsqueeze(-1) ** 2))

        y = (coords[:, :, 0] - mean[:, 1].unsqueeze(-1)) ** 2
        y = y / (self._beta * (std[:, 1].unsqueeze(-1) ** 2))

        heatmaps = torch.exp(-(x + y))  # (B, HW)

        return heatmaps

    def __call__(self, boxes: BBoxes, size: tuple[int, int]) -> Float[Tensor, "B HW"]:
        return super.__call__(boxes, size)  # type: ignore


class SpatialRelationEncodings(nn.Module):
    def __init__(self, hidden_dim: int, temperature: int = 10000) -> None:
        super().__init__()

        # center coordinates x 2 = 4
        # width and height x 2 = 4
        # area x 2 = 2
        # aspect ratio x 2 = 2
        # intersection over union = 1
        # union area = 1
        # center distance x 2 = 2
        # total = 16

        if hidden_dim % 16 != 0:
            raise ValueError("hidden_dim must be divisible by 16.")

        self._dim = hidden_dim // 16
        self._temperature = temperature

    def forward(self, boxes1: BBoxes, boxes2: BBoxes) -> Float[Tensor, "B C"]:
        boxes1 = boxes1.normalize()
        boxes2 = boxes2.normalize()

        cxcywh1 = boxes1.to_cxcywh()  # (B, 4)
        cxcywh2 = boxes2.to_cxcywh()  # (B, 4)
        xyxy1 = boxes1.to_xyxy()  # (B, 4)
        xyxy2 = boxes2.to_xyxy()  # (B, 4)

        center1 = cxcywh1.tensor[:, :2]  # (B, 2)
        center2 = cxcywh2.tensor[:, :2]  # (B, 2)
        wh1 = cxcywh1.tensor[:, 2:]  # (B, 2)
        wh2 = cxcywh2.tensor[:, 2:]  # (B, 2)

        area1 = wh1.prod(dim=1, keepdim=True)  # (B, 1)
        area2 = wh2.prod(dim=1, keepdim=True)  # (B, 1)

        aspect_ratio1 = (wh1[:, 0] / wh1[:, 1])[None]  # (B, 1)
        aspect_ratio2 = (wh2[:, 0] / wh2[:, 1])[None]  # (B, 1)

        iou, union = ops.box_iou_pairwise(xyxy1.tensor, xyxy2.tensor)  # (B,)
        iou = iou[None]  # (B, 1)
        union = union[None]  # (B, 1)

        distancex = (center1[:, 0] - center2[:, 0])[None]  # (B, 1)
        distancey = (center1[:, 1] - center2[:, 1])[None]  # (B, 1)

        pos = torch.cat(
            (
                center1,
                center2,
                wh1,
                wh2,
                area1,
                area2,
                aspect_ratio1,
                aspect_ratio2,
                iou,
                union,
                distancex,
                distancey,
            ),
        )  # (B, 16)
        pos = pos.unsqueeze(-1)  # (B, 16, 1)

        # transform this vector into a sinuoidal encoding
        dim_t = torch.arange(
            self._dim, dtype=torch.float32, device=boxes1.device
        )  # (C / 16)
        dim_t = self._temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self._dim
        )  # (C / 16)

        pos = pos / dim_t  # (B, 16, C / 16)
        pos[..., 0::2] = pos[..., 0::2].sin()
        pos[..., 1::2] = pos[..., 1::2].cos()

        return pos.view(-1, self._dim * 16)  # (B, C)

    def __call__(self, boxes1: BBoxes, boxes2: BBoxes) -> Float[Tensor, "B C"]:
        return super.__call__(boxes1, boxes2)  # type: ignore
