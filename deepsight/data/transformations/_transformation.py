##
##
##


import abc

from deepsight.data.structs import BoundingBoxes, Image


class Transformation(abc.ABC):
    """Base class for all transformations."""

    @abc.abstractmethod
    def __call__(
        self, image: Image, boxes: BoundingBoxes | None = None
    ) -> tuple[Image, BoundingBoxes | None]:
        """Applies the transformation to the image and bounding boxes
        (if any).

        Parameters
        ----------
        image : Image
            The image to transform.
        boxes : BoundingBoxes, optional
            The bounding boxes to transform, by default None

        Returns
        -------
        tuple[Image, BoundingBoxes | None]
            The transformed image and bounding boxes (if any).
        """
        ...
