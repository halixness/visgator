##
##
##

from jaxtyping import Float
from torch import Tensor

from deepsight.data.structs import BoundingBoxes
from deepsight.measures import Reduction

from ._utils import reduce_loss


def generalized_box_iou_loss(
    predictions: BoundingBoxes,
    targets: BoundingBoxes,
    reduction: Reduction = Reduction.MEAN,
) -> Float[Tensor, "..."]:
    """Computes the pairwise generalized IoU [1]_ loss between the predicted and target
    boxes.

    Parameters
    ----------
    predictions : BoundingBoxes
        The predicted boxes.
    targets : BoundingBoxes
        The corresponding target boxes. Must have the same shape as `predictions`.
    reduction : Reduction, optional
        The reduction type. Defaults to Reduction.MEAN.

    Returns
    -------
    Float[Tensor, "..."]
        The computed generalized IoU for each pair of boxes.

    References
    ----------
    .. [1] Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., & Savarese, S.
        (2019). Generalized intersection over union: A metric and a loss for
        bounding box regression. In Proceedings of the IEEE/CVF conference on computer
        vision and pattern recognition (pp. 658-666).
    """

    loss = 1 - predictions.generalized_iou(targets)
    return reduce_loss(loss, reduction)
