##
##
##

from jaxtyping import Float
from torch import Tensor

from deepsight.data.structs import BoundingBoxes
from deepsight.measures import Reduction

from ._utils import reduce_loss


def box_iou_loss(
    predictions: BoundingBoxes,
    targets: BoundingBoxes,
    reduction: Reduction = Reduction.MEAN,
) -> Float[Tensor, "..."]:
    """Computes the pairwise IoU loss between the predicted and target boxes.

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
        A tensor containing the IoU values. If `reduction` is Reduction.NONE, the
        tensor will have shape (*N,) where *N is the number of leading dimensions
        of `predictions`. Otherwise, the tensor will be a scalar.
    """

    loss = 1 - predictions.iou(targets)
    return reduce_loss(loss, reduction)
