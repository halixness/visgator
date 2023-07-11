##
##
##

from dataclasses import dataclass
from typing import Iterable

import torch

from deepsight.data.structs import BoundingBoxes, Image

from ._transformation import Transformation


@dataclass(init=False, frozen=True, slots=True)
class Compose(Transformation):
    """Composes multiple transformations into one.

    Parameters
    ----------
    transformations : tuple[Transformation, ...]
        The transformations to compose.
    p : float, optional
        The probability to start applying the transformations. Defaults to 1.0.
    """

    transformations: tuple[Transformation, ...]
    p: float = 1.0

    def __init__(
        self, transformations: Iterable[Transformation], p: float = 1.0
    ) -> None:
        object.__setattr__(self, "transformations", tuple(transformations))
        object.__setattr__(self, "p", p)

    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        if torch.rand(1) > self.p:
            return image, boxes

        for transformation in self.transformations:
            image, boxes = transformation(image, boxes)

        return image, boxes
