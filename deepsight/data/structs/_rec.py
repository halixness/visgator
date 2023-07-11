##
##
##

from dataclasses import dataclass

import torch
from typing_extensions import Self

from deepsight.utils.protocols import Moveable

from ._boxes import BoundingBoxes
from ._image import TensorImage


@dataclass(frozen=True, slots=True)
class RECInput(Moveable):
    """Dataclass representing an input sample for referring expression
    comprehension.

    Attributes
    ----------
    image : Image
        The input image.
    description : str
        The description of the region of interest in the input image.
    """

    image: TensorImage
    description: str

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

        return self.__class__(self.image.to(device), self.description)


@dataclass(frozen=True, slots=True)
class RECOutput(Moveable):
    """Dataclass storing the output of referring expression comprehension
    for a single input sample.

    Attributes
    ----------
    box : BoundingBoxes
        The bounding box of the region in the input image. The bounding box is
        represented as a tensor of shape (4,).
    """

    box: BoundingBoxes

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

        return self.__class__(self.box.to(device))
