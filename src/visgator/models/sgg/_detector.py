##
##
##

import torch
import torchvision.transforms as T
from groundingdino.models import GroundingDINO
from groundingdino.util.inference import load_model
from groundingdino.util.misc import NestedTensor
from groundingdino.util.utils import get_phrases_from_posmap
from torch import nn

from visgator.utils.batch import Batch
from visgator.utils.bbox import BBoxes, BBoxFormat
from visgator.utils.misc import Nested4DTensor

from ._config import DetectorConfig
from ._misc import DetectionResults


class Detector(nn.Module):
    def __init__(self, config: DetectorConfig) -> None:
        super().__init__()

        self._transform = T.Compose(
            [
                T.Lambda(lambda x: x / 255),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self._gdino: GroundingDINO = load_model(str(config.config), str(config.weights))
        for param in self._gdino.parameters():
            param.requires_grad = False

        self._box_threshold = config.box_threshold
        self._text_threshold = config.text_threshold

    def forward(self, batch: Batch) -> list[DetectionResults]:
        images = Nested4DTensor.from_tensors(
            [self._transform(sample.image) for sample in batch]
        )

        entities: list[list[str]] = [None] * len(batch)  # type: ignore
        captions: list[str] = [None] * len(batch)  # type: ignore
        for i, sample in enumerate(batch):
            graph = sample.caption.graph
            assert graph is not None

            entities[i] = [entity.head.lower().strip() for entity in graph.entities]
            captions[i] = " . ".join(entities[i]) + " ."

        gdino_images = NestedTensor(images.tensor, images.mask)
        output = self._gdino(gdino_images, captions=captions)

        pred_logits = output["pred_logits"].sigmoid()
        pred_boxes = output["pred_boxes"]

        masks = pred_logits.max(dim=2)[0] > self._box_threshold

        detections: list[DetectionResults] = [None] * len(batch)  # type: ignore
        for i in range(len(batch)):
            mask = masks[i]
            detected_boxes = pred_boxes[i, mask]
            logits = pred_logits[i, mask] > self._text_threshold
            tokenized = self._gdino.tokenizer(captions[i])

            phrases: list[str] = [
                get_phrases_from_posmap(logit, tokenized, self._gdino.tokenizer)
                for logit in logits
            ]

            detected_entities: dict[str, list[int]] = {}
            for idx, phrase in enumerate(phrases):
                detected_entities.setdefault(phrase, []).append(idx)

            indexes = []
            boxes = []

            for idx, entity in enumerate(entities[i]):
                if entity in detected_entities:
                    for j in detected_entities[entity]:
                        indexes.append(j)
                        boxes.append(detected_boxes[j])
                else:
                    raise RuntimeError(f"Entity {entity} not detected.")

            detections[i] = DetectionResults(
                entities=torch.tensor(indexes),
                boxes=BBoxes(
                    boxes=torch.stack(boxes),
                    images_size=images.sizes[i],
                    format=BBoxFormat.XYXY,
                    normalized=True,
                ),
            )

        return detections

    def __call__(self, batch: Batch) -> list[DetectionResults]:
        return super().__call__(batch)  # type: ignore
