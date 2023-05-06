##
##
##

from typing import Optional

import torchvision.transforms as T
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO

from visgator.utils.batch import Batch
from visgator.utils.bbox import BBox, BBoxes

from .._criterion import Criterion
from .._model import Model as BaseModel
from ._config import Config


class Model(BaseModel[BBoxes]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._yolo = YOLO(config.yolo.weights())
        self._clip_processor = CLIPProcessor.from_pretrained(config.clip.weights())
        self._clip = CLIPModel.from_pretrained(config.clip.weights())

        self._toPIL = T.ToPILImage()

    @property
    def criterion(self) -> Optional[Criterion[BBoxes]]:
        return None

    def predict(self, output: BBoxes) -> BBoxes:
        return output

    def forward(self, batch: Batch) -> BBoxes:
        boxes = BBoxes()

        images = [self._toPIL(sample.image) for sample in batch.samples]
        yolo_results = self._yolo.predict(images, conf=0.5, verbose=False)
        for sample, result in zip(batch, yolo_results):
            proposals = []

            if len(result.boxes) == 0:
                # create a dummy bbox
                tmp = BBox.from_tuple(
                    (0, 0, 0, 0),
                    sample.image.shape[1:],  # type: ignore
                )
                boxes.append(tmp.to(self._clip.device))
                continue

            for bbox in result.boxes:
                xmin, ymin, xmax, ymax = bbox.xyxy.int()[0]
                clipped_image = sample.image[:, ymin:ymax, xmin:xmax]
                proposals.append(clipped_image)

            inputs = self._clip_processor(
                text=sample.sentence,
                images=proposals,
                return_tensors="pt",
            ).to(self._clip.device)

            output = self._clip(**inputs)
            best = output.logits_per_image.argmax(0).item()
            boxes.append(
                BBox(
                    result.boxes[best].xyxy[0],
                    sample.image.shape[1:],  # type: ignore
                )
            )

        return boxes
