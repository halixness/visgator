##
##
##

import abc
from typing import Generic, TypeVar

from torch import nn

from deepsight.data.structs import Batch
from deepsight.utils.protocols import Moveable

T = TypeVar("T", bound=Moveable)
U = TypeVar("U", bound=Moveable)
V = TypeVar("V")


class PreProcessor(abc.ABC, Generic[T, U, V], nn.Module):
    """Base class for all pre-processors.

    Pre-processors are used to transform the input data before it is passed to the
    model. For example, a pre-processor can be used to resize the input images to
    the size expected by the model or to normalize the input data.

    At training time, the pre-processor is called with both the input data and the
    targets. This is useful for training strategies that require the targets, such
    as denoising strategies. At inference time, the pre-processor is called with
    only the input data.
    """

    @abc.abstractmethod
    def forward(self, inputs: Batch[T], targets: Batch[U] | None) -> V:
        ...

    def __call__(self, inputs: Batch[T], targets: Batch[U] | None) -> V:
        return super().__call__(inputs, targets)  # type: ignore
