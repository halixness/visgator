##
##
##

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Shaped
from torch import Tensor
from typing_extensions import Self


@dataclass(init=False, frozen=True, slots=True)
class Batched2DTensors:
    """Dataclass to store a batch of 2D tensors with different lengths in a single 3D
    tensor.

    Attributes
    ----------
    tensor : Shaped[Tensor, "B L D"]
        The 3D tensor, where B is the batch size, L is the maximum length, and D is
        the number of features.
    mask : Bool[Tensor, "B L"]
        The mask of the 3D tensor, where True indicates a padded value.
    sizes : list[int]
        The lengths of the 2D tensors.
    """

    # -------------------------------------------------------------------------
    # Attributes and properties
    # -------------------------------------------------------------------------

    tensor: Shaped[Tensor, "B L D"]
    mask: Bool[Tensor, "B L"]
    sizes: list[int]

    @property
    def device(self) -> torch.device:
        """The device of the tensor."""
        return self.tensor.device

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the tensor."""
        return self.tensor.dtype

    @property
    def shape(self) -> torch.Size:
        """The shape of the tensor."""
        return self.tensor.shape

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(
        self,
        tensor: Shaped[Tensor, "B L D"],
        mask: Bool[Tensor, "B L"] | None = None,
        sizes: list[int] | None = None,
    ) -> None:
        if mask is None and sizes is None:
            raise ValueError("Expected either mask or sizes to be provided.")

        if mask is None:
            mask = tensor.new_ones(tensor.shape[:2], dtype=torch.bool)
            for idx, size in enumerate(sizes):  # type: ignore
                mask[idx, :size] = False

        if sizes is None:
            sizes = (~mask).sum(dim=-1).tolist()

        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "sizes", sizes)

    @classmethod
    def from_list(
        cls,
        tensors: list[Shaped[Tensor, "L D"]],
        pad_value: float = 0.0,
    ) -> Self:
        """Creates a batch of 2D tensors from a list of 2D tensors.

        Parameters
        ----------
        tensors : list[Shaped[Tensor, "L D"]]
            A list of 2D tensors.
        pad_value : float, optional
            The value to pad the tensors with, by default 0.0.

        Returns
        -------
        Batched2DTensors
            The batch of 2D tensors.
        """

        sizes = [tensor.shape[0] for tensor in tensors]
        B = len(tensors)
        L = max(sizes)
        D = tensors[0].shape[1]

        tensor = torch.full(
            (B, L, D),
            fill_value=pad_value,
            device=tensors[0].device,
            dtype=tensors[0].dtype,
        )

        for idx, size in enumerate(sizes):
            tensor[idx, :size].copy_(tensors[idx])

        return cls(tensor, None, sizes)

    # -------------------------------------------------------------------------
    # Other methods
    # -------------------------------------------------------------------------

    def to_list(self) -> list[Shaped[Tensor, "L D"]]:
        """Converts the Batched2DTensors to a list of 2D tensors without padding.

        Returns
        -------
        list[Shaped[Tensor, "L D"]]
            A list of 2D tensors.
        """

        return [self.tensor[idx, :size] for idx, size in enumerate(self.sizes)]

    # -------------------------------------------------------------------------
    # Magic methods
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Returns the number of 2D tensors.

        Returns
        -------
        int
            The number of 2D tensors.
        """
        return len(self.sizes)
