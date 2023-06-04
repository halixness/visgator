##
##
##

from typing import Optional

from typing_extensions import Self

from visgator.models import Model as _Model
from visgator.utils.batch import Batch
from visgator.utils.bbox import BBoxes, BBoxFormat

from ._config import Config
from ._criterion import Criterion
from ._decoder import Decoder
from ._detector import Detector
from ._encoders import build_encoders
from ._head import RegressionHead
from ._misc import Graph, NestedGraph
from ._postprocessor import PostProcessor


class Model(_Model[BBoxes]):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self._criterion = Criterion(config.criterion)
        self._postprocessor = PostProcessor()

        self._detector = Detector(config.detector)
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
        detections = self._detector(batch)
        img_embeddings = self._vision(batch)
        text_embeddings = self._text(batch)

        graphs = [
            Graph.new(batch.samples[idx].caption, text_embeddings[idx], detections[idx])
            for idx in range(len(batch))
        ]
        graph = NestedGraph.from_graphs(graphs)

        B = len(graph)  # B
        N = max([nodes for nodes, _ in graph.sizes])  # N

        padded_boxes = detections[0].boxes.tensor.new_zeros(B * N, 4)
        images_size = detections[0].boxes.images_size.new_ones(B * N, 2)

        for idx in range(len(graph)):
            boxes = detections[idx].boxes.to_cxcywh().normalize()
            start = idx * N
            end = start + boxes.tensor.shape[0]

            padded_boxes[start:end] = boxes.tensor
            images_size[start:end] = boxes.images_size

        boxes = BBoxes(padded_boxes, images_size, BBoxFormat.CXCYWH, True)  # (BN, 4)
        graph = self._decoder(img_embeddings, graph, boxes)

        return self._head(graph, img_embeddings.sizes)
