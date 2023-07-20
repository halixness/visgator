##
##
##

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor
from typing_extensions import Self

from deepsight.utils.protocols import Moveable

from ._boxes import BoundingBoxes
from ._image import TensorImage


@dataclass(frozen=True, slots=True)
class ODInput(Moveable):
    """Dataclass representing an input sample for object detection.

    Attributes
    ----------
    image : Image
        The input image.
    entities : list[str]
        The entities that should be detected in the input image if the model
        supports open-set object detection. If the model does not support
        open-set object detection, this attribute is ignored.
    """

    image: TensorImage
    entities: list[str]

    def to(self, device: torch.device | str) -> Self:
        """Moves the input to the given device.

        Parameters
        ----------
        device : torch.device | str
            The device to move the input to.

        Returns
        -------
        Self
            The input moved to the given device.
        """

        return self.__class__(self.image.to(device), self.entities)


@dataclass(frozen=True, slots=True)
class ODOutput(Moveable):
    """Dataclass storing the output of object detection.

    Attributes
    ----------
    boxes : BoundingBoxes
        The ground truth bounding boxes of the entities in the input sample.
        The bounding boxes are represented as a tensor of shape (N, 4) where
        N is the number of entities in the input sample.
    entities : Int[Tensor, "N"]
        If the model supports open-set object detection, this attribute is a
        tensor containing the index of the entity in the input sample that each
        bounding box corresponds to. If the model does not support open-set
        object detection, this attribute is a tensor containing the index of the
        category that each bounding box corresponds to.
    scores : Float[Tensor, "N"]
        The confidence score of each bounding box.
    """

    boxes: BoundingBoxes
    entities: Int[Tensor, "N"]  # noqa: F821
    scores: Float[Tensor, "N"]  # noqa: F821

    def to(self, device: torch.device | str) -> Self:
        """Moves the target to the given device.

        Parameters
        ----------
        device : torch.device | str
            The device to move the target to.

        Returns
        -------
        Self
            The target moved to the given device.
        """

        return self.__class__(
            self.boxes.to(device),
            self.entities.to(device),
            self.scores.to(device),
        )
