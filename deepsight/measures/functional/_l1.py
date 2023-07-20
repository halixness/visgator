##
##
##

import torch
from jaxtyping import Float
from torch import Tensor

from deepsight.data.structs import BoundingBoxes, BoundingBoxFormat
from deepsight.measures import Reduction

from ._utils import reduce_loss


def box_l1_loss(
    predictions: BoundingBoxes,
    targets: BoundingBoxes,
    format: BoundingBoxFormat | None = BoundingBoxFormat.CXCYWH,
    normalized: bool | None = True,
    reduction: Reduction = Reduction.MEAN,
) -> Float[Tensor, "..."]:
    """Computes the pairwise L1 loss between the predicted and target boxes.

    Parameters
    ----------
    predictions : BoundingBoxes
        The predicted boxes.
    targets : BoundingBoxes
        The corresponding target boxes. Must have the same shape as `predictions`.
    format : BoundingBoxFormat, optional
        The format of the boxes in which to transform the boxes before computing the
        loss. If None, the boxes will not be transformed and the format in which they
        are provided will be used. Defaults to BoundingBoxFormat.CXCYWH.
    normalized : bool, optional
        Whether to use normalized coordinates. If None, the coordinates will not be
        normalized and the format in which they are provided will be used. Defaults to
        True.
    reduction : Reduction, optional
        The reduction type. Defaults to Reduction.MEAN.

    Returns
    -------
    Float[Tensor, "..."]
        A tensor containing the L1 loss values. If `reduction` is Reduction.NONE, the
        tensor will have shape (*N,) where *N is the number of leading dimensions
        of `predictions`. Otherwise, the tensor will be a scalar.
    """

    if normalized is True:
        predictions = predictions.normalize()
        targets = targets.normalize()
    elif normalized is False:
        predictions = predictions.denormalize()
        targets = targets.denormalize()

    if format is not None:
        predictions = predictions.to_format(format)
        targets = targets.to_format(format)

    predictions.pairwise_check(targets)

    loss = torch.cdist(predictions.tensor, targets.tensor, p=1)
    loss = loss.diagonal(dim1=-1, dim2=-2)
    return reduce_loss(loss, reduction)
