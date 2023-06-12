##
##
##


import torch
import torchvision.transforms as T
from torch import Tensor, nn
from transformers import OwlViTProcessor, OwlViTForObjectDetection

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

        self._toPIL = T.ToPILImage()
        self._box_threshold = config.box_threshold
        self._text_threshold = config.text_threshold
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._txt_similarity_threshold = 0.5
        self._detection_threshold = 0.2

        self._detector_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", torch_dtype=torch.float16)
        self._detector_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", torch_dtype=torch.float16)


    def forward(self, data: tuple, model: tuple) -> list[DetectionResults]:
        # partialy taken from: https://huggingface.co/docs/transformers/model_doc/owlvit
        
        batch, nested_images = data
        clip, tokenizer = model

        # Preprocessing & YOLO
        images = [self._toPIL(sample.image) for sample in batch.samples]
        captions = [sample.caption for sample in batch.samples]
        
        B = len(captions)

        # Extracting graph entities
        entities: list[list[str]] = [None] * B  # type: ignore
        for i, caption in enumerate(captions):
            graph = caption.graph
            assert graph is not None
            entities[i] = [entity.head.lower().strip() for entity in graph.entities]

        # Object detection (open-vocabulary)
        inputs = self._detector_processor(text=entities, images=images, return_tensors="pt").to(self.device)
        detector_results = self._detector_model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size for image in images]).to(self.device)
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self._detector_processor.post_process_object_detection(outputs=detector_results, target_sizes=target_sizes)

        # For each result
        detections: list[DetectionResults] = [None] * B  # type: ignore

        for sample_idx in range(B):
            boxes, scores, labels = results[sample_idx]["boxes"], results[sample_idx]["scores"], results[sample_idx]["labels"]
            
            matched_indices = []
            matched_boxes = []

            # Check identified identities by score
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                if score >= self._detection_threshold:
                    matched_boxes.append(box)
                    matched_indices.append(label) # entity index

            raise Exception(torch.tensor(matched_boxes).shape)

            detections[sample_idx] = DetectionResults(
                entities=torch.tensor(matched_indices, device=self.device, dtype=torch.int),
                boxes=BBoxes(
                    boxes=torch.tensor(matched_boxes),
                    images_size=images[sample_idx].size,
                    format=BBoxFormat.XYXY,  # CXCYWH
                    normalized=False, # TODO: check
                ),
            )

        return detections
    

    def __call__(
        self, images: Nested4DTensor, captions: list[Caption]
    ) -> list[DetectionResults]:
        return super().__call__(images, captions)  # type: ignore
