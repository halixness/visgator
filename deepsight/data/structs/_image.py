##
##
##

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torchvision
import torchvision.transforms.functional as F
from jaxtyping import Num
from PIL import Image as pImage
from torch import Tensor
from typing_extensions import Self

from deepsight.utils.protocols import Moveable


@dataclass(frozen=True, slots=True)
class TensorImage(Moveable):
    """A wrapper around a tensor image."""

    data: Num[Tensor, "3 H W"]
    normalized: bool

    @property
    def size(self) -> tuple[int, int]:
        """The size of the image.

        Returns
        -------
        tuple[int, int]
            The size of the image.
        """

        return (self.data.shape[1], self.data.shape[2])

    @classmethod
    def open(self, path: Path | str) -> Self:
        data = torchvision.io.read_image(str(path))
        if data.ndim == 2:
            data = data.unsqueeze(0)

        if data.shape[0] == 1:
            data = data.repeat(3, 1, 1)

        return self(data=data, normalized=False)

    def normalize(self) -> Self:
        """Normalizes the image to [0, 1].

        Returns
        -------
        TensorImage
            The normalized image.
        """

        if self.normalized:
            return self

        return self.__class__(data=self.data / 255, normalized=True)

    def denormalize(self) -> Self:
        """Denormalizes the image to [0, 255].

        Returns
        -------
        TensorImage
            The denormalized image.
        """

        if not self.normalized:
            return self

        data = self.data * 255
        data = data.to(torch.uint8)
        return self.__class__(data=data, normalized=False)

    def to_tensor(self) -> Self:
        """Converts the image to a tensor.

        Returns
        -------
        TensorImage
            The converted image.
        """

        return self

    def to_pil(self) -> PILImage:
        """Converts the image to a PIL image.

        Returns
        -------
        PILImage
            The converted image.
        """

        data = F.to_pil_image(self.data)
        return PILImage(data=data)

    def to_numpy(self) -> NumpyImage:
        """Converts the image to a numpy image.

        Returns
        -------
        NumpyImage
            The converted image.
        """

        image = self.denormalize()
        data = image.data.cpu().numpy()
        return NumpyImage(data=data)

    def to(self, device: torch.device | str) -> Self:
        """Moves the image to the given device.

        Parameters
        ----------
        device : torch.device | str
            The device to move the image to.

        Returns
        -------
        TensorImage
            The moved image.
        """

        return self.__class__(data=self.data.to(device), normalized=self.normalized)


@dataclass(frozen=True, slots=True)
class PILImage:
    """A wrapper around a PIL image."""

    data: pImage.Image

    @property
    def size(self) -> tuple[int, int]:
        """The size of the image.

        .. note::
            Unlike the property `size` of the PIL Image, this property returns
            the size in the format (height, width).

        Returns
        -------
        tuple[int, int]
            The size of the image.
        """

        return (self.data.height, self.data.width)

    @classmethod
    def open(self, path: Path | str) -> Self:
        """Opens an image from the given path.

        .. note::
            The image is converted to RGB.

        Parameters
        ----------
        path : Path
            The path to the image.

        Returns
        -------
        PILImage
            The opened image.
        """

        return TensorImage.open(path).to_pil()  # type: ignore

    def to_tensor(self) -> TensorImage:
        """Converts the image to a tensor.

        Returns
        -------
        TensorImage
            The converted image.
        """

        data = F.to_tensor(self.data)
        return TensorImage(data=data, normalized=False)

    def to_pil(self) -> Self:
        """Converts the image to a PIL image.

        Returns
        -------
        PILImage
            The converted image.
        """

        return self

    def to_numpy(self) -> NumpyImage:
        """Converts the image to a numpy image.

        Returns
        -------
        NumpyImage
            The converted image.
        """

        data = np.asarray(self.data)
        return NumpyImage(data=data)


@dataclass(frozen=True, slots=True)
class NumpyImage:
    """A wrapper around an image stored as a numpy array."""

    data: npt.NDArray[np.uint8]

    @property
    def size(self) -> tuple[int, int]:
        """The size of the image.

        Returns
        -------
        tuple[int, int]
            The size of the image.
        """

        return (self.data.shape[0], self.data.shape[1])

    @classmethod
    def open(self, path: Path | str) -> Self:
        """Opens an image from the given path.

        Parameters
        ----------
        path : Path
            The path to the image.

        Returns
        -------
        OpenCVImage
            The opened image.
        """

        return TensorImage.open(path).to_numpy()  # type: ignore

    def to_tensor(self) -> TensorImage:
        """Converts the image to a tensor.

        Returns
        -------
        TensorImage
            The converted image.
        """

        data = F.to_tensor(self.data)
        return TensorImage(data=data, normalized=False)

    def to_pil(self) -> PILImage:
        """Converts the image to a PIL image.

        Returns
        -------
        PILImage
            The converted image.
        """

        data = F.to_pil_image(self.data)
        return PILImage(data=data)

    def to_numpy(self) -> Self:
        """Converts the image to a numpy image.

        Returns
        -------
        NumpyImage
            The converted image.
        """

        return self


Image = TensorImage | PILImage | NumpyImage
