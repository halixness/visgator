##
##
##

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from deepsight.data.structs import BoundingBoxes


class GaussianHeatmaps(nn.Module):
    """Gaussian heatmaps.

    This module computes a Gaussian heatmap for a set of bounding boxes.
    The gaussian is centered at the center of the bounding box and its standard
    devation in each direction is equal to the corresponding side of the bounding
    box.

    These heatmaps can be used to bias the attention of each query to a specific region
    of the image during the cross-attention step of DETR-like models. See [1]_ for more
    details.

    Attributes
    ----------
    beta : float
        A scaling factor for the gaussian standard deviation. The smaller the value,
        the more concentrated the gaussian will be around the center of the bounding
        box. Defaults to 1.0.

    References
    ----------
    .. [1] Gao, P., Zheng, M., Wang, X., Dai, J. and Li, H., 2021. Fast convergence of
        detr with spatially modulated co-attention. In Proceedings of the IEEE/CVF
        international conference on computer vision (pp. 3621-3630).
    """

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()

        self.beta = beta

    def forward(
        self,
        boxes: BoundingBoxes,
        mask: Bool[Tensor, "... H W"],
    ) -> Float[Tensor, "... H W"]:
        boxes = boxes.to_cxcywh().normalize()

        mean = boxes.tensor[..., :2]  # (..., 2)
        std = boxes.tensor[..., 2:]  # (..., 2)

        not_mask = ~mask
        y_coords = not_mask.cumsum(dim=-2, dtype=torch.float32)  # (..., H, W)
        x_coords = not_mask.cumsum(dim=-1, dtype=torch.float32)  # (..., H, W)

        eps = torch.finfo(torch.float32).eps
        y_coords = y_coords / (y_coords[..., -1:, :] + eps)
        x_coords = x_coords / (x_coords[..., -1:] + eps)

        y = (y_coords - mean[..., 1, None, None]) ** 2
        y = y / (self.beta * (std[..., 1, None, None] ** 2))

        x = (x_coords - mean[..., 0, None, None]) ** 2
        x = x / (self.beta * (std[..., 0, None, None] ** 2))

        out: Tensor = torch.exp(-(x + y))  # (..., H, W)
        out.masked_fill_(mask, 0.0)

        return out

    def __call__(
        self,
        boxes: BoundingBoxes,
        mask: Bool[Tensor, "... H W"],
    ) -> Float[Tensor, "... H W"]:
        """Computes log-Gaussian heatmaps.

        Parameters
        ----------
        boxes : BoundingBoxes
            The bounding boxes for which to compute the heatmaps. The bounding boxes
            tensor can have any number of leading dimensions.
        mask : Bool[Tensor, "... H W"]
            A boolean mask indicating which pixels in the heatmaps should be considered
            as padding, i.e. which pixels are outside the image. The mask tensor must
            have the same number of leading dimensions as the bounding boxes tensor.

        Returns
        -------
        Float[Tensor, "... H W"]
            The computed heatmaps. The number of leading dimensions of the returned
            tensor is equal to the number of leading dimensions of the bounding boxes
            tensor. The last two dimensions are equal to the height and width of the
            heatmaps, respectively.
        """

        return super().__call__(boxes, mask)  # type: ignore
