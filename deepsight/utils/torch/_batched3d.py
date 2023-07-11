##
##
##

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Shaped
from torch import Tensor
from typing_extensions import Self

from ._batched2d import Batched2DTensors


@dataclass(init=False, frozen=True, slots=True)
class Batched3DTensors:
    """Dataclass to store a batch of 3D tensors with different heights and widths
    in a single 4D tensor.

    Attributes
    ----------
    tensor : Shaped[Tensor, "B C H W"]
        The 4D tensor, where B is the batch size, C is the number of channels,
        H is the maximum height, and W is the maximum width.
    mask : Bool[Tensor, "B H W"]
        The mask of the 4D tensor, where True indicates a padded value.
    sizes : list[tuple[int, int]]
        The heights and widths of the 3D tensors.
    """

    # -------------------------------------------------------------------------
    # Attributes and properties
    # -------------------------------------------------------------------------

    tensor: Shaped[Tensor, "B C H W"]
    mask: Bool[Tensor, "B H W"]
    sizes: list[tuple[int, int]]

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
        tensor: Shaped[Tensor, "B C H W"],
        mask: Bool[Tensor, "B H W"] | None = None,
        sizes: list[tuple[int, int]] | None = None,
    ) -> None:
        if mask is None and sizes is None:
            raise ValueError("Expected either mask or sizes to be provided.")

        if mask is None:
            mask = tensor.new_ones(
                (tensor.shape[0], tensor.shape[2], tensor.shape[3]),
                dtype=torch.bool,
            )

            for idx, (height, width) in enumerate(sizes):  # type: ignore
                mask[idx, :height, :width] = False

        if sizes is None:
            not_mask = ~mask
            sizes = [
                (
                    int(not_mask[idx].sum(dim=0).max()),
                    int(not_mask[idx].sum(dim=1).max()),
                )
                for idx in range(not_mask.shape[0])
            ]

        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "sizes", sizes)

    @classmethod
    def from_list(
        cls,
        tensors: list[Shaped[Tensor, "C H W"]],
        pad_value: float = 0.0,
    ) -> Self:
        """Creates a batch of 3D tensors from a list of 3D tensors.

        Parameters
        ----------
        tensors : list[Shaped[Tensor, "C H W"]]
            The list of 3D tensors.
        pad_value : float, optional
            The value to pad the tensors with, by default 0.0.

        Returns
        -------
        Batched3DTensors
            The batch of 3D tensors.
        """

        sizes = [(tensor.shape[1], tensor.shape[2]) for tensor in tensors]
        max_height = max(size[0] for size in sizes)
        max_width = max(size[1] for size in sizes)

        tensor = torch.full(
            (len(tensors), tensors[0].shape[0], max_height, max_width),
            pad_value,
            dtype=tensors[0].dtype,
            device=tensors[0].device,
        )

        for idx, (height, width) in enumerate(sizes):
            tensor[idx, :, :height, :width].copy_(tensors[idx])

        return cls(tensor=tensor, sizes=sizes)

    # -------------------------------------------------------------------------
    # Other methods
    # -------------------------------------------------------------------------

    def to_list(self) -> list[Shaped[Tensor, "C H W"]]:
        """Converts the Batched3DTensors to a list of 3D tensors without padding.

        Returns
        -------
        list[Shaped[Tensor, "C H W"]]
            The list of 3D tensors.
        """

        return [
            self.tensor[idx, :, :height, :width]
            for idx, (height, width) in enumerate(self.sizes)
        ]

    def to_batched2d(self) -> Batched2DTensors:
        """Converts the Batched3DTensors to a Batched2DTensors.

        Returns
        -------
        Batched2DTensors
            The Batched2DTensors.
        """

        # (B, C, H, W) -> (B, C, H * W) -> (B, H * W, C)
        tensor = self.tensor.flatten(2).transpose(1, 2)
        mask = self.mask.flatten(1)  # (B, H, W) -> (B, H * W)
        sizes = [h * w for h, w in self.sizes]

        return Batched2DTensors(tensor=tensor, mask=mask, sizes=sizes)

    # -------------------------------------------------------------------------
    # Magic methods
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Returns the number of 3D tensors."""
        return len(self.sizes)
