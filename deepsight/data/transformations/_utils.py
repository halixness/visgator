##
##
##

import albumentations as A
import torch

from deepsight.data.structs import BoundingBoxes, BoundingBoxFormat, Image, NumpyImage


def check_boxes(boxes: BoundingBoxes, image_size: tuple[int, int]) -> None:
    """Checks if the bounding boxes belong to the image, i.e. if the bounding boxes
    have the same size as the image.

    Parameters
    ----------
    boxes : BoundingBoxes
        The bounding boxes.
    image_size : tuple[int, int]
        The size of the image.

    Raises
    ------
    ValueError
        If any of the bounding boxes does not belong to the image.
    """

    height, width = image_size
    images_sizes = boxes.images_size

    if not torch.all(images_sizes[..., 0] == height) or not torch.all(
        images_sizes[..., 1] == width
    ):
        raise ValueError("The bounding boxes do not belong to the image.")


def apply_albumentation_transform(
    transform: A.Compose, image: Image, boxes: BoundingBoxes | None
) -> tuple[Image, BoundingBoxes | None]:
    image = image.to_numpy()

    if boxes is not None:
        check_boxes(boxes, image.size)
        boxes_list = boxes_to_albumentations(boxes)

        transformed = transform(image=image, bboxes=boxes_list)
        transformed_image = transformed["image"]
        transformed_boxes = transformed["bboxes"]

        image = NumpyImage(transformed_image)
        boxes = albumentations_to_boxes(transformed_boxes, boxes, image.size)
    else:
        transformed = transform(image=image)
        transformed_image = transformed["image"]

        image = NumpyImage(transformed_image)

    return image, boxes

    ...


def boxes_to_albumentations(boxes: BoundingBoxes) -> list[list[float]]:
    """Transforms the bounding boxes to the format expected by the Albumentations,
    i.e. a list of lists of floats.

    All the bounding boxes are converted to a list of lists of floats, even if the
    bounding boxes tensor has only one bounding box. They are also converted to the
    pascal_voc format, i.e. [xmin, ymin, xmax, ymax].

    Parameters
    ----------
    boxes : BoundingBoxes
        The bounding boxes.

    Returns
    -------
    list[list[float]]
        The bounding boxes in the format expected by the Albumentations.

    Raises
    ------
    ValueError
        If the bounding boxes tensor has more than 2 dimensions.
    """

    if boxes.tensor.ndim > 2:
        raise ValueError("The bounding boxes tensor must have at most 2 dimensions.")

    boxes = boxes.to_xyxy().denormalize()

    if boxes.tensor.ndim == 1:
        tensor = boxes.tensor.unsqueeze(0)
    else:
        tensor = boxes.tensor

    return tensor.tolist()


def albumentations_to_boxes(
    boxes_list: list[list[float]],
    boxes: BoundingBoxes,
    new_image_size: tuple[int, int],
) -> BoundingBoxes:
    """Transforms the bounding boxes from the format expected by the Albumentations
    to the format expected by BoundingBoxes.

    The boxes in the provided list are assumed to be in the pascal_voc format, i.e.
    [xmin, ymin, xmax, ymax].

    Parameters
    ----------
    boxes_list : list[list[float]]
        The bounding boxes in the format expected by the Albumentations.
    boxes : BoundingBoxes
        The original bounding boxes from which the bounding boxes in the list were
        extracted.
    new_image_size : tuple[int, int]
        The new size of the image to which the bounding boxes belong.

    Returns
    -------
    BoundingBoxes
        The bounding boxes in the format expected by BoundingBoxes.
    """

    new_boxes: list[float] | list[list[float]]
    if boxes.tensor.ndim == 1:
        new_boxes = boxes_list[0]
    else:
        new_boxes = boxes_list

    tensor = torch.tensor(new_boxes, device=boxes.device, dtype=boxes.tensor.dtype)
    return BoundingBoxes(
        tensor,
        new_image_size,
        format=BoundingBoxFormat.XYXY,
        normalized=False,
    )
