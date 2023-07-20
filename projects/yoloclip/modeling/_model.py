##
##
##

import torch
from torch import Tensor, nn
from transformers.models.clip import CLIPModel, CLIPProcessor

from deepsight.data.structs import Batch, BoundingBoxes, ODInput, RECInput, RECOutput
from deepsight.modeling.detectors import YOLO
from deepsight.modeling.pipeline import Model as _Model

from ._config import Config


class Model(_Model[Batch[RECInput], Batch[RECOutput]]):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self._dummy = nn.Parameter(torch.empty(0))

        self.yolo = YOLO(config.yolo, config.box_threshold)

        self.clip = CLIPModel.from_pretrained(config.clip.weights())
        self.processor = CLIPProcessor.from_pretrained(config.clip.weights())

    def forward(self, inputs: Batch[RECInput]) -> Batch[RECOutput]:
        det_results = self.yolo(Batch([ODInput(inp.image, []) for inp in inputs]))

        outputs = []
        for sample_idx, result in enumerate(det_results):
            if len(result.entities) == 0:
                outputs.append(RECOutput(result.boxes))
                continue

            image = inputs[sample_idx].image.denormalize().data
            cropped_regions: list[Tensor] = []
            boxes = result.boxes.to_xyxy().denormalize()
            for bbox in boxes.tensor:
                x1, y1, x2, y2 = bbox.int()
                cropped_regions.append(image[:, y1:y2, x1:x2])

            tmp = self.processor(
                text=inputs[sample_idx].description,
                images=cropped_regions,
                return_tensors="pt",
            ).to(self._dummy.device)

            output = self.clip(**tmp)
            idx = output.logits_per_image.argmax(0).item()

            outputs.append(
                RECOutput(
                    BoundingBoxes(
                        boxes.tensor[idx],
                        images_size=boxes.images_size[idx],
                        format=boxes.format,
                        normalized=boxes.normalized,
                    )
                )
            )

        return Batch(outputs)
