##
##
##

import enum
from dataclasses import dataclass
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor


class Reduction(enum.Enum):
    """Enum class for the reduction type of a loss."""

    NONE = "none"
    SUM = "sum"
    MEAN = "mean"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Loss:
    """Dataclass storing loss information.

    Attributes
    ----------
    name : str
        Name of the loss.
    value : Float[Tensor, ""]
        Tensor containing the loss value.
    weight : float
        Weight of the loss to be used in the total loss.
    """

    name: str
    value: Float[Tensor, ""]
    weight: float


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Losses(Generic[T]):
    """Dataclass storing loss information.

    Attributes
    ----------
    total : T
        Information about the total loss. The total loss is obtained by summing
        the scaled losses.
    unscaled : dict[str, T]
        Dictionary containing the information for each unscaled loss. The unscaled
        losses are the ones provided by the criterion.
    scaled : dict[str, T]
        Dictionary containing the information for each scaled loss. The scaled
        losses are the ones provided by the criterion multiplied by the weight
        of the loss.
    """

    total: T
    unscaled: dict[str, T]
    scaled: dict[str, T]

    def flatten(self) -> list[tuple[str, T]]:
        """Returns a list of tuples containing the name and value of each loss.

        The name of the total loss is "total". The name of the unscaled losses
        are prefixed with "unscaled/". The name of the scaled losses are prefixed
        with "scaled/".

        Returns
        -------
        list[tuple[str, T]]
            List of tuples containing the name and value of each loss.
        """

        return [
            ("total", self.total),
            *[(f"unscaled/{name}", value) for name, value in self.unscaled.items()],
            *[(f"scaled/{name}", value) for name, value in self.scaled.items()],
        ]
