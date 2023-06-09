##
##
##

from typing import Optional

from typing_extensions import Self

from visgator.models import Model as _Model
from visgator.utils.batch import Batch
from visgator.utils.bbox import BBoxes, BBoxFormat
from visgator.utils.torch import Nested4DTensor
from visgator.utils.transforms import Compose, Resize

from ._config import Config
from ._criterion import Criterion
from ._decoder import Decoder
from ._detector import Detector
from ._encoders import build_encoders
from ._head import RegressionHead
from ._misc import Graph, NestedGraph
from ._postprocessor import PostProcessor

import torch

class Model(_Model[BBoxes]):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self._transform = Compose(
            [Resize(800, max_size=1333, p=1.0)],
            p=1.0,
        )

        self._criterion = Criterion(config.criterion)
        self._postprocessor = PostProcessor()

        self._detector = Detector(config.detector)
        self._vision, self._text, self._model, self._tokenizer = build_encoders(config.encoders, return_model=True)
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
        
        """
        # Grounding DINO bboxes
         detections = self._detector(
            images, [sample.caption for sample in batch.samples]
        )
        """

        detections = self._detector(
            (batch, images),
            (self._model, self._tokenizer)
        )

        # CLIP encoded img+text
        img_embeddings = self._vision(images)
        text_embeddings = self._text(batch)

        # Constructing the batch graphs with entity embeddings
        graphs = [
            Graph.new(batch.samples[idx].caption, text_embeddings[idx], detections[idx])
            for idx in range(len(batch))
        ]
        graph = NestedGraph.from_graphs(graphs)

        B = len(graph)  # B
        N = max([nodes for nodes, _ in graph.sizes])  # N

        padded_boxes = detections[0].boxes.tensor.new_ones(B * N, 4) # TODO: setting them to zeros causes nans
        images_size = detections[0].boxes.images_size.new_ones(B * N, 2)

        for idx in range(len(graph)):
            boxes = detections[idx].boxes.to_cxcywh().normalize()
            start = idx * N
            end = start + boxes.tensor.shape[0]

            padded_boxes[start:end] = boxes.tensor
            images_size[start:end] = boxes.images_size

        # Free up space
        del detections
        del text_embeddings
        torch.cuda.empty_cache()

        # ERP-Cross Attention (img+text)
        boxes = BBoxes(padded_boxes, images_size, BBoxFormat.CXCYWH, True)  # (BN, 4)
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
