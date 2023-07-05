##
##
##

from typing import Optional

from typing_extensions import Self

from visgator.models import Model as _Model
from visgator.utils.batch import Batch
from visgator.utils.bbox import BBoxes
from visgator.utils.torch import Nested4DTensor
from visgator.utils.transforms import Compose, Resize

from ._config import Config
from ._criterion import Criterion
from ._decoder import Decoder
from ._encoders import build_encoders
from ._gdino import GroundingDINODetector
from ._head import RegressionHead
from ._misc import Graph, pad_sequences
from ._owlvit import OwlViTDetector
from ._postprocessor import PostProcessor


class Model(_Model[BBoxes]):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self._transform = Compose(
            [Resize(800, max_size=1333, p=1.0)],
            p=1.0,
        )

        self._criterion = Criterion(config.criterion)
        self._postprocessor = PostProcessor()

        self._gdino = None
        self._owlvit = None
        if config.detector.gdino is not None:
            self._gdino = GroundingDINODetector(config.detector)
        elif config.detector.owlvit is not None:
            self._owlvit = OwlViTDetector(config.detector)

        self._vision, self._text = build_encoders(config.encoders)
        self._decoder = Decoder(config.decoder)

        self._head = RegressionHead(config.head)

    @classmethod
    def from_config(cls, config: Config) -> Self:  # type: ignore
        return cls(config)

    @property
    def name(self) -> str:
        return "ERPA"

    @property
    def criterion(self) -> Optional[Criterion]:
        return self._criterion

    @property
    def postprocessor(self) -> PostProcessor:
        return self._postprocessor

    def forward(self, batch: Batch) -> BBoxes:
        images = Nested4DTensor.from_tensors(
            [self._transform(sample.image) for sample in batch.samples]
        )
        img_tensor = images.tensor / 255.0
        images = Nested4DTensor(img_tensor, images.sizes, images.mask)

        if self._gdino is not None:
            detections = self._gdino(
                images, [sample.caption for sample in batch.samples]
            )
        elif self._owlvit is not None:
            detections = self._owlvit(batch)
        else:
            raise RuntimeError("No detector is initialized.")

        # CLIP encoded img+text
        img_embeddings = self._vision(images)
        text_embeddings = self._text(batch)

        # Constructing the batch graphs with entity embeddings
        graphs = [
            Graph.new(batch.samples[idx].caption, text_embeddings[idx], detections[idx])
            for idx in range(len(batch))
        ]

        boxes, graph = pad_sequences(detections, graphs)
        graph = self._decoder(img_embeddings, graph, boxes)

        # ERP-Cross Attention (img+text)
        boxes = self._head(graph, img_embeddings.sizes)
        if not boxes.normalized:
            raise RuntimeError("Boxes must be normalized.")

        boxes = BBoxes(
            boxes.tensor,
            [tuple(sample.image.shape[1:]) for sample in batch],  # type: ignore
            boxes.format,
            True,
        )

        return boxes
