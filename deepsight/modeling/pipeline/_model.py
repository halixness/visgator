##
##
##

import abc
from typing import Generic, TypeVar

from torch import nn

T = TypeVar("T")
U = TypeVar("U")


class Model(abc.ABC, Generic[T, U], nn.Module):
    """Base class for all models.

    Models take the pre-processed input data and produce the model output. The model
    output is then passed to the post-processor to produce the final output or to the
    criterion to compute the losses.
    """

    @abc.abstractmethod
    def forward(self, inputs: T) -> U:
        ...

    def __call__(self, inputs: T) -> U:
        return super().__call__(inputs)  # type: ignore
