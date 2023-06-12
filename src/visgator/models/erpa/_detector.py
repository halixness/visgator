##
##
##


import torch
import torchvision.transforms as T
from torch import Tensor, nn
from ultralytics import YOLO

from visgator.utils.batch import Caption
from visgator.utils.bbox import BBoxes, BBoxFormat
from visgator.utils.torch import Nested4DTensor

from ._config import DetectorConfig
from ._misc import DetectionResults


class Detector(nn.Module):
    def __init__(self, config: DetectorConfig) -> None:
        super().__init__()

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1))
        self._mean: Tensor  # just for type hinting
        self._std: Tensor  # just for type hinting

        """
        self._gdino: GroundingDINO = load_model(str(config.config), str(config.weights))
        for param in self._gdino.parameters():
            param.requires_grad = False
        """

        self._yolo = YOLO(config.yolo.weights())
        self._toPIL = T.ToPILImage()

        self._box_threshold = config.box_threshold
        self._text_threshold = config.text_threshold
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._txt_similarity_threshold = 0.5

    def forward(self, data: tuple, model: tuple) -> list[DetectionResults]:
        batch, nested_images = data
        clip, tokenizer = model

        # Preprocessing & YOLO
        images = [self._toPIL(sample.image) for sample in batch.samples]
        captions = [sample.caption for sample in batch.samples]
        yolo_results = self._yolo.predict(images, conf=0.5, verbose=False)

        B = len(captions)

        # Extracting graph entities
        # TODO: problem here when no entities are found
        entities: list[list[str]] = [None] * B  # type: ignore
        for i, caption in enumerate(captions):
            graph = caption.graph
            assert graph is not None
            entities[i] = [entity.head.lower().strip() for entity in graph.entities]

        # Matching yolo detected
        detections: list[DetectionResults] = [None] * B  # type: ignore
        for sample_idx, sample_result in enumerate(yolo_results):  # B
            detected_entities: dict[str, list[int]] = {}
            detected_boxes = sample_result.boxes.xyxy

            # Identify unique type of entities
            for idx, cls in enumerate(sample_result.boxes.cls):
                cls = sample_result.names[int(cls)]
                detected_entities.setdefault(cls, []).append(idx)

            indexes = []
            boxes = []

            # Match detected entities with caption ones
            for entity_idx, entity in enumerate(entities[sample_idx]):
                # CLIP-fashioned entity-detections matching!
                entity_emb = clip.encode_text(tokenizer(entity).to(self.device))
                entity_emb /= entity_emb.norm(dim=-1, keepdim=True)

                detected_ents_emb = clip.encode_text(
                    tokenizer(list(detected_entities.keys())).to(self.device)
                )
                detected_ents_emb /= detected_ents_emb.norm(dim=-1, keepdim=True)

                scores = (100.0 * entity_emb @ detected_ents_emb.T).softmax(dim=-1)[0]
                match_idx = torch.argmax(scores).item()
                match_key = list(detected_entities.keys())[match_idx]

                if scores[match_idx] > self._txt_similarity_threshold:
                    for detection_idx in detected_entities[match_key]:
                        indexes.append(entity_idx)
                        boxes.append(detected_boxes[detection_idx])

                else:  # TODO: undetected
                    pass

            detections[sample_idx] = DetectionResults(
                entities=torch.tensor(indexes, device=self.device),
                boxes=BBoxes(
                    boxes=torch.stack(boxes),
                    images_size=nested_images.sizes[sample_idx],
                    format=BBoxFormat.XYXY,  # CXCYWH
                    normalized=True,
                ),
            )

        return detections

    """
    def forward(
        self, images: Nested4DTensor, captions: list[Caption]
    ) -> list[DetectionResults]:

        img_tensor = (images.tensor - self._mean) / self._std
        img_tensor.masked_fill_(images.mask.unsqueeze(1).expand(-1, 3, -1, -1), 0.0)
        images = Nested4DTensor(img_tensor, images.sizes, images.mask)
        B = len(captions)

        entities: list[list[str]] = [None] * B  # type: ignore
        sentences: list[str] = [None] * B  # type: ignore
        for i, caption in enumerate(captions):
            graph = caption.graph
            assert graph is not None

            entities[i] = [entity.head.lower().strip() for entity in graph.entities]
            sentences[i] = " . ".join(entities[i]) + " ."

        gdino_images = NestedTensor(images.tensor, images.mask)

        output = self._gdino(gdino_images, captions=sentences)

        pred_logits = output["pred_logits"].sigmoid()
        pred_boxes = output["pred_boxes"]

        masks = pred_logits.max(dim=2)[0] > self._box_threshold

        detections: list[DetectionResults] = [None] * B  # type: ignore
        for sample_idx in range(B):
            mask = masks[sample_idx] # B, 900
            detected_boxes = pred_boxes[sample_idx, mask] # K, 4
            logits = pred_logits[sample_idx, mask] > self._text_threshold # K, 256
            tokenized = self._gdino.tokenizer(sentences[sample_idx]) # one sentence per batch

            # repeats the same phrase for each bounding box detected
            phrases: list[str] = [
                get_phrases_from_posmap(logit, tokenized, self._gdino.tokenizer)
                for logit in logits
            ] # B,

            detected_entities: dict[str, list[int]] = {}
            for idx, phrase in enumerate(phrases):
                detected_entities.setdefault(phrase, []).append(idx)

            indexes = []
            boxes = []

            for entity_idx, entity in enumerate(entities[sample_idx]):

                matching_key = None
                for key in detected_entities.keys():
                    if entity in key: matching_key = key

                if matching_key is not None:
                    for detection_idx in detected_entities[matching_key]:
                        indexes.append(entity_idx)
                        boxes.append(detected_boxes[detection_idx])
                else:
                    # this can happen when the box/text threshold is too high
                    # raise RuntimeError(f"Entity {entity} not detected.")
                    pass

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
    """

    def __call__(
        self, images: Nested4DTensor, captions: list[Caption]
    ) -> list[DetectionResults]:
        return super().__call__(images, captions)  # type: ignore
