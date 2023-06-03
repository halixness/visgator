##
##
##

from typing import Optional

import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor
from typing_extensions import Self

from visgator.models import Model as _Model
from visgator.utils.batch import Batch
from visgator.utils.bbox import BBox, BBoxes, BBoxFormat

from ._config import Config
from ._criterion import Criterion
from ._postprocessor import PostProcessor


class Model(_Model[BBoxes]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._postprocessor = PostProcessor()

        self._processor = CLIPProcessor.from_pretrained(config.clip.weights())
        self._clip = CLIPModel.from_pretrained(config.clip.weights())
        output_dim = (
            self._clip.config.vision_config.projection_dim
            + self._clip.config.text_config.projection_dim
        )
        self._head = nn.Sequential(nn.Linear(output_dim, 4), nn.Sigmoid())

        for param in self._clip.parameters():
            param.requires_grad = False

    @classmethod
    def from_config(cls, config: Config) -> Self:  # type: ignore
        return cls(config)

    @property
    def criterion(self) -> Optional[Criterion]:
        return Criterion()

    @property
    def postprocessor(self) -> PostProcessor:
        return self._postprocessor

    def forward(self, batch: Batch) -> BBoxes:
        images = [sample.image for sample in batch]
        sentences = [sample.caption.sentence for sample in batch]

        inputs = self._processor(
            text=sentences,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self._clip.device)

        clip_output = self._clip(
            **inputs,
            output_attentions=False,
            output_hidden_states=False,
            return_loss=False,
            return_dict=True,
        )

        image_embeds = clip_output.image_embeds
        text_embeds = clip_output.text_embeds

        embeds = torch.cat((image_embeds, text_embeds), dim=-1)
        outputs = self._head(embeds)

        bboxes = []
        for sample, output in zip(batch, outputs):
            bboxes.append(BBox(output, sample.image.shape[1:], BBoxFormat.CXCYWH, True))

        return BBoxes.from_bboxes(bboxes)
