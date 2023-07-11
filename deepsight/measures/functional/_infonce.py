##
##
##

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from deepsight.measures import Reduction


def infonce_loss(
    queries: Float[Tensor, "N D"],
    positive_keys: Float[Tensor, "N D"],
    negative_keys: Float[Tensor, "M D"],
    temperature: float = 0.1,
    reduction: Reduction = Reduction.MEAN,
) -> Float[Tensor, ""] | Float[Tensor, "N"]:  # noqa: F821
    """Computes the InfoNCE loss.

    .. note::
        At the moment, this function only supports unpaired mode where the queries
        are contrasted with all the negative keys.

    Parameters
    ----------
    queries : Float[Tensor, "N D"]
        Tensor containing the queries.
    positive_keys : Float[Tensor, "N D"]
        Tensor containing the positive keys.
    negative_keys : Float[Tensor, "M D"]
        Tensor containing the negative keys.
    temperature : float, optional
        Temperature of the loss. Defaults to 0.1.
    reduction : Reduction, optional
        Reduction type of the loss. Defaults to Reduction.MEAN.

    Returns
    -------
    Float[Tensor, ""] | Float[Tensor, "N"]
        Tensor containing the loss value. If the reduction is Reduction.NONE, the
        tensor has shape (N,). Otherwise, it is a scalar.
    """

    B = queries.shape[0]

    queries = F.normalize(queries, dim=-1)  # (B, D)
    positive_keys = F.normalize(positive_keys, dim=-1)  # (B, D)
    negative_keys = F.normalize(negative_keys, dim=-1)  # (M, D)

    pos_logits = torch.sum(queries * positive_keys, dim=-1, keepdim=True)  # (B, 1)
    neg_logits = queries @ negative_keys.T  # (B, M)

    logits = torch.cat((pos_logits, neg_logits), dim=-1)  # (B, 1+M)
    logits /= temperature  # (B, 1+M)

    labels = torch.zeros(B, dtype=torch.long, device=logits.device)  # (B,)
    info_nce_loss = F.cross_entropy(logits, labels, reduction=str(reduction))

    return info_nce_loss
