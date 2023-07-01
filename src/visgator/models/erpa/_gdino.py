##
##
##

import bisect

import torch
import torchvision.transforms.functional as F
from groundingdino.models import GroundingDINO
from groundingdino.util.inference import load_model
from groundingdino.util.misc import NestedTensor
from torch import Tensor, nn
from transformers import AutoTokenizer

from visgator.utils.batch import Caption
from visgator.utils.bbox import BBoxes, BBoxFormat
from visgator.utils.torch import Nested4DTensor

from ._config import DetectorConfig
from ._misc import DetectionResults


class GroundigDINODetector(nn.Module):
    def __init__(self, config: DetectorConfig) -> None:
        super().__init__()

        assert config.gdino is not None

        self._mean = (0.485, 0.456, 0.406)
        self._std = (0.229, 0.224, 0.225)

        self._gdino: GroundingDINO = load_model(
            str(config.gdino.config), str(config.gdino.weights)
        )

        self._box_threshold = config.box_threshold
        self._text_threshold = config.text_threshold
        self._max_detections = config.max_detections

        self._freeze()

    def _freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def _get_phrases_from_posmap(
        self,
        posmap: torch.BoolTensor,
        tokenized: dict[str, Tensor],
        tokenizer: AutoTokenizer,
        left_idx: int = 0,
        right_idx: int = 255,
    ) -> str:
        assert isinstance(posmap, torch.Tensor), "posmap must be torch.Tensor"
        if posmap.dim() != 1:
            raise ValueError("posmap must be 1-dim")

        posmap[0 : left_idx + 1] = False
        posmap[right_idx:] = False
        non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        return tokenizer.decode(token_ids)  # type: ignore

    def forward(
        self, images: Nested4DTensor, captions: list[Caption]
    ) -> list[DetectionResults]:
        img_tensor = F.normalize(images.tensor, self._mean, self._std)
        img_tensor.masked_fill_(images.mask.unsqueeze(1).expand(-1, 3, -1, -1), 0.0)
        images = Nested4DTensor(img_tensor, images.sizes, images.mask)
        B = len(captions)

        entities: list[dict[str, list[int]]] = [{} for _ in range(B)]
        sentences: list[str] = [""] * B
        for i, caption in enumerate(captions):
            graph = caption.graph
            assert graph is not None

            for entity_idx, entity in enumerate(graph.entities):
                head = entity.head.lower().strip()
                entities[i].setdefault(head, []).append(entity_idx)

            sentences[i] = " . ".join(entities[i].keys()) + " ."
        del i

        gdino_images = NestedTensor(images.tensor, images.mask)
        output = self._gdino(gdino_images, captions=sentences)

        pred_logits = output["pred_logits"].sigmoid()
        pred_boxes = output["pred_boxes"]

        masks = pred_logits.max(dim=2)[0] > self._box_threshold

        detections: list[DetectionResults] = [None] * B  # type: ignore
        for sample_idx in range(B):
            mask = masks[sample_idx]
            detected_boxes = pred_boxes[sample_idx, mask]
            logits = pred_logits[sample_idx, mask]

            if len(logits) > self._max_detections:
                logits, indices = torch.topk(logits, self._max_detections)
                detected_boxes = detected_boxes[indices]

            tokenized = self._gdino.tokenizer(sentences[sample_idx])

            sep_idx = [
                i
                for i in range(len(tokenized["input_ids"]))
                if tokenized["input_ids"][i] in [101, 102, 1012]
            ]

            phrases: list[str] = []
            for logit in logits:
                max_idx = logit.argmax()
                insert_idx = bisect.bisect_left(sep_idx, max_idx)
                right_idx = sep_idx[insert_idx]
                left_idx = sep_idx[insert_idx - 1]
                phrases.append(
                    self._get_phrases_from_posmap(
                        logit > self._text_threshold,
                        tokenized,
                        self._gdino.tokenizer,
                        left_idx,
                        right_idx,
                    ).replace(".", "")
                )

            indexes = []
            boxes = []
            entities_found: list[bool] = [False] * len(
                captions[sample_idx].graph.entities  # type: ignore
            )

            for det_idx, det_name in enumerate(phrases):
                if det_name in entities[sample_idx]:
                    for entity_idx in entities[sample_idx][det_name]:
                        indexes.append(entity_idx)
                        boxes.append(detected_boxes[det_idx])
                        entities_found[entity_idx] = True
                else:
                    for entity_name, entity_idxs in entities[sample_idx].items():
                        if det_name in entity_name:
                            for entity_idx in entity_idxs:
                                indexes.append(entity_idx)
                                boxes.append(detected_boxes[det_idx])
                                entities_found[entity_idx] = True

            for entity_idx, found in enumerate(entities_found):
                if not found:
                    indexes.append(entity_idx)
                    boxes.append(torch.tensor([0.5, 0.5, 0.5, 0.5]))

            detections[sample_idx] = DetectionResults(
                entities=torch.tensor(indexes, device=boxes[0].device),
                boxes=BBoxes(
                    boxes=torch.stack(boxes),
                    images_size=images.sizes[sample_idx],
                    format=BBoxFormat.CXCYWH,
                    normalized=True,
                ),
            )

        return detections

    def __call__(
        self, images: Nested4DTensor, captions: list[Caption]
    ) -> list[DetectionResults]:
        return super().__call__(images, captions)  # type: ignore
