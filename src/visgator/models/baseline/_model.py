##
##
##

from typing import Optional

import open_clip
import torch
import torchvision.transforms as T
from torch import nn
from typing_extensions import Self

from visgator.models import Model as _Model
from visgator.utils.batch import Batch
from visgator.utils.bbox import BBox, BBoxes, BBoxFormat

from ._config import Config
from ._criterion import Criterion
from ._postprocessor import PostProcessor


class Model(_Model[BBoxes]):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self._postprocessor = PostProcessor()

        self._dummy = nn.Parameter(torch.empty(0))

        model, _, preprocess = open_clip.create_model_and_transforms(
            config.model,
            config.pretrained,
        )
        self._clip = model
        self._img_preprocess = preprocess
        self._toPIL = T.ToPILImage()

        self._tokenizer = open_clip.get_tokenizer(config.model)

        output_dim = self._clip.text_projection.shape[1]
        self._head = nn.Sequential(nn.Linear(2 * output_dim, 4), nn.Sigmoid())

        for param in self._clip.parameters():
            param.requires_grad = False

    @classmethod
    def from_config(cls, config: Config) -> Self:  # type: ignore
        return cls(config)

    @property
    def name(self) -> str:
        return "Baseline"

    @property
    def criterion(self) -> Optional[Criterion]:
        return Criterion()

    @property
    def postprocessor(self) -> PostProcessor:
        return self._postprocessor

    def forward(self, batch: Batch) -> BBoxes:
        images = [self._img_preprocess(self._toPIL(sample.image)) for sample in batch]
        sentences = [sample.caption.sentence for sample in batch]

        image = torch.stack(images).to(self._dummy.device)
        text = self._tokenizer(sentences).to(self._dummy.device)

        image_embeds = self._clip.encode_image(image)
        text_embeds = self._clip.encode_text(text)

        embeds = torch.cat((image_embeds, text_embeds), dim=-1)
        outputs = self._head(embeds)

        bboxes = []
        for sample, output in zip(batch, outputs):
            bboxes.append(BBox(output, sample.image.shape[1:], BBoxFormat.CXCYWH, True))

        return BBoxes.from_bboxes(bboxes)
