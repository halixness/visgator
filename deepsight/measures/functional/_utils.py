##
##
##

from jaxtyping import Float
from torch import Tensor

from deepsight.measures import Reduction


def reduce_loss(
    loss: Float[Tensor, "..."], reduction: Reduction
) -> Float[Tensor, "..."]:
    """Reduces a loss tensor according to the given reduction type.

    Parameters
    ----------
    loss : Float[Tensor, "..."]
        The loss tensor.
    reduction : Reduction
        The reduction type.

    Returns
    -------
    Float[Tensor, "..."]
        The reduced loss tensor, or the original loss tensor if `reduction` is
        Reduction.NONE.
    """

    match reduction:
        case Reduction.NONE:
            return loss
        case Reduction.SUM:
            return loss.sum()
        case Reduction.MEAN:
            return loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        case _:
            raise ValueError(f"Invalid reduction type: {reduction}")
