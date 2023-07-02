##
##
##

import torch
import torchvision.transforms.functional as T
from torch import nn
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from visgator.utils.batch import Batch
from visgator.utils.bbox import BBoxes, BBoxFormat

from ._config import DetectorConfig
from ._misc import DetectionResults


class OwlViTDetector(nn.Module):
    def __init__(self, config: DetectorConfig) -> None:
        super().__init__()

        assert config.owlvit is not None

        self._dummy = nn.Parameter(torch.empty(0))

        self._box_threshold = config.box_threshold
        self._max_detections = config.max_detections

        self._preprocessor = OwlViTProcessor.from_pretrained(config.owlvit)
        self._detector = OwlViTForObjectDetection.from_pretrained(config.owlvit)

        self._freeze()

    def _freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, batch: Batch) -> list[DetectionResults]:
        # partialy taken from: https://huggingface.co/docs/transformers/model_doc/owlvit

        # Preprocessing
        images = [T.to_pil_image(sample.image) for sample in batch.samples]
        captions = [sample.caption for sample in batch.samples]

        B = len(captions)

        # Extracting graph entities
        entities: list[list[str]] = [None] * B  # type: ignore
        for i, caption in enumerate(captions):
            graph = caption.graph
            assert graph is not None
            entities[i] = [
                f"a photo of {entity.head.lower().strip()}" for entity in graph.entities
            ]

        # Object detection (open-vocabulary)
        inputs = self._preprocessor(
            text=entities, images=images, return_tensors="pt"
        ).to(self._dummy.device)
        detector_results = self._detector(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.tensor(
            [image.size for image in images], device=self._dummy.device
        )
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self._preprocessor.post_process_object_detection(
            outputs=detector_results, target_sizes=target_sizes
        )

        # For each result
        detections: list[DetectionResults] = [None] * B  # type: ignore

        for sample_idx in range(B):
            boxes, scores, labels = (
                results[sample_idx]["boxes"],
                results[sample_idx]["scores"],
                results[sample_idx]["labels"],
            )

            matched_indices = []
            matched_boxes = []
            height, width = images[sample_idx].size

            entities_found = [False] * len(entities[sample_idx])

            idx = scores >= self._box_threshold
            boxes = boxes[idx]
            scores = scores[idx]
            labels = labels[idx]

            # If detections are too many => select tok K first
            if len(boxes) > self._max_detections:
                _, idx = torch.topk(scores, self._max_detections)
                boxes = boxes[idx]
                scores = scores[idx]
                labels = labels[idx]

            # Check identified identities by score
            for box, label in zip(boxes, labels):
                matched_boxes.append(box)
                matched_indices.append(label)
                entities_found[label] = True

            for entity_idx, found in enumerate(entities_found):
                if not found:
                    matched_indices.append(entity_idx)
                    matched_boxes.append(
                        torch.tensor([0.0, 0.0, width - 1, height - 1]).to(
                            self._dummy.device
                        )
                    )

            boxes = BBoxes(
                boxes=torch.stack(matched_boxes),
                images_size=images[sample_idx].size,
                format=BBoxFormat.XYXY,
                normalized=False,
            )

            detections[sample_idx] = DetectionResults(
                entities=torch.tensor(matched_indices, device=self._dummy.device),
                boxes=boxes,
            )

        del inputs
        del detector_results
        del target_sizes
        del results
        # torch.cuda.empty_cache()

        return detections

    def __call__(self, batch: Batch) -> list[DetectionResults]:
        return super().__call__(batch)  # type: ignore
