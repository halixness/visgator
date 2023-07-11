##
##
##

from typing import Protocol

import torch
from typing_extensions import Self


class Moveable(Protocol):
    """A protocol for objects that can be moved to a device."""

    def to(self, device: torch.device | str) -> Self:
        """Moves the object to the given device.

        Parameters
        ----------
        device : torch.device | str
            The device to move the object to.

        Returns
        -------
        Self
            The object moved to the given device.
        """
        ...
