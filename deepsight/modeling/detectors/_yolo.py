##
##
##

import enum

import torch
import ultralytics
from torch import nn
from typing_extensions import Self

from deepsight.data.structs import (
    Batch,
    BoundingBoxes,
    BoundingBoxFormat,
    ODInput,
    ODOutput,
)


class YOLOModel(enum.Enum):
    NANO = "n"
    SMALL = "s"
    MEDIUM = "m"
    LARGE = "l"
    EXTRA = "x"

    @classmethod
    def from_str(cls, s: str) -> Self:
        return cls[s.upper()]

    def weights(self) -> str:
        return f"yolov8{self.value}.pt"


class YOLO(nn.Module):
    """This class wraps the YOLO model from Ultralytics."""

    def __init__(self, model: YOLOModel, confidence: float = 0.25):
        super().__init__()

        self.model = ultralytics.YOLO(model.weights())
        self.confidence = confidence

    def forward(self, inputs: Batch[ODInput]) -> Batch[ODOutput]:
        images = [inp.image.to_pil().data for inp in inputs]
        results = self.model.predict(images, conf=self.confidence, verbose=False)

        outputs = []
        for sample_idx, result in enumerate(results):
            if len(result.boxes) > 0:
                boxes = BoundingBoxes(
                    result.boxes.xyxyn,
                    images_size=inputs[sample_idx].image.size,
                    format=BoundingBoxFormat.XYXY,
                    normalized=True,
                )

                indices = result.boxes.data
                scores = result.boxes.conf

                outputs.append(ODOutput(boxes, indices, scores))
            else:
                # here we create bbox since BoundingBoxes requires at least one bbox
                boxes = BoundingBoxes(
                    torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.model.device),
                    images_size=inputs[sample_idx].image.size,
                    format=BoundingBoxFormat.XYXY,
                    normalized=True,
                )

                # however, we don't want to return any indices or scores
                indices = torch.tensor([], device=self.model.device)
                scores = torch.tensor([], device=self.model.device)

                outputs.append(ODOutput(boxes, indices, scores))

        return Batch(outputs)

    def __call__(self, inputs: Batch[ODInput]) -> Batch[ODOutput]:
        """ "Given a batch of images, returns the detected objects.

        .. note::
            If no objects are detected in an image, the returned bounding boxes for the
            corresponding image will contain a dummy bounding box with all coordinates
            set to 0.0. This is due to the fact that `BoundingBoxes` requires at least
            one bounding box to be present. However, the returned indices and scores
            will be empty tensors.
        """

        return super().__call__(inputs)  # type: ignore
