##
##
##

import abc
from typing import Generic, TypeVar

from torch import nn

from deepsight.data.structs import Batch
from deepsight.measures import Loss
from deepsight.utils.protocols import Moveable

T = TypeVar("T")
U = TypeVar("U", bound=Moveable)


class Criterion(abc.ABC, Generic[T, U], nn.Module):
    """Base class for all criteria.

    Criteria are used to compute the loss from the model output and the targets.
    """

    @abc.abstractmethod
    def losses_names(self) -> list[str]:
        """Returns the names of the losses computed by the criterion."""
        ...

    @abc.abstractmethod
    def forward(self, output: T, targets: Batch[U]) -> list[Loss]:
        ...

    def __call__(self, output: T, targets: Batch[U]) -> list[Loss]:
        """Computes the loss from the model output and the targets.

        Parameters
        ----------
        output : T
            Model output.
        targets : Batch[Target]
            Targets.

        Returns
        -------
        list[Loss]
            List of losses.
        """
        return super().__call__(output, targets)  # type: ignore
