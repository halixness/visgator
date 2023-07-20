##
##
##

from typing import Generic, Iterable, Iterator, TypeVar, overload

import torch
from typing_extensions import Self

from deepsight.utils.protocols import Moveable

T = TypeVar("T", bound=Moveable)


class Batch(Moveable, Generic[T]):
    """Class representing a batch of samples."""

    def __init__(self, elements: Iterable[T] | Iterator[T]) -> None:
        super().__init__()

        self._elements = tuple(elements)

    def to(self, device: torch.device | str) -> Self:
        """Moves the batch to the given device.

        Parameters
        ----------
        device : torch.device | str
            The device to move the batch to.

        Returns
        -------
        Self
            The batch moved to the given device.
        """

        return self.__class__(element.to(device) for element in self._elements)

    # ---------------------------------------------------------------------- #
    # Magic Methods
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._elements)

    def __iter__(self) -> Iterator[T]:
        return iter(self._elements)

    @overload
    def __getitem__(self, index: int) -> T:
        ...

    @overload
    def __getitem__(self, index: slice) -> Self:
        ...

    def __getitem__(self, index: int | slice) -> T | Self:
        if isinstance(index, int):
            return self._elements[index]

        return self.__class__(self._elements[index])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self._elements}"

    def __str__(self) -> str:
        return self.__repr__()
