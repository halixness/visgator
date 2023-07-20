##
##
##

import abc
from typing import Generic, TypeVar

from torch import nn

from deepsight.data.structs import Batch
from deepsight.utils.protocols import Moveable

T = TypeVar("T")
U = TypeVar("U", bound=Moveable)


class PostProcessor(abc.ABC, Generic[T, U], nn.Module):
    """Base class for all post-processors.

    Post-processors are used to transform the model output into the final output.
    """

    @abc.abstractmethod
    def forward(self, output: T) -> Batch[U]:
        ...

    def __call__(self, output: T) -> Batch[U]:
        return super().__call__(output)  # type: ignore
