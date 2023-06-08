##
##
##

from typing import Optional

import torch
from torch import nn
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
from ._misc import Graph, ModelOutput, NestedGraph
from ._postprocessor import PostProcessor


class Model(_Model[ModelOutput]):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self._transform = Compose(
            [Resize(800, max_size=1333, p=1.0)],
            p=1.0,
        )

        self._criterion = Criterion(config.criterion)
        self._postprocessor = PostProcessor()

        self._detector = Detector(config.detector)
        self._vision, self._text = build_encoders(config.encoders)
        self._decoder = Decoder(config.decoder)
        self._same_entity_edge = nn.Parameter(torch.randn(1, config.decoder.hidden_dim))

        self._proj = nn.Linear(config.head.hidden_dim, config.head.hidden_dim)
        self._regression_head = nn.Sequential(
            nn.Linear(config.head.hidden_dim, config.head.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.head.dropout),
            nn.Linear(config.head.hidden_dim, 4),
        )

    @classmethod
    def from_config(cls, config: Config) -> Self:  # type: ignore
        return cls(config)

    @property
    def name(self) -> str:
        return "SceneGraphGrounder"

    @property
    def criterion(self) -> Optional[Criterion]:
        return self._criterion

    @property
    def postprocessor(self) -> PostProcessor:
        return self._postprocessor

    def forward(self, batch: Batch) -> ModelOutput:
        images = Nested4DTensor.from_tensors(
            [self._transform(sample.image) for sample in batch.samples]
        )
        img_tensor = images.tensor / 255.0
        images = Nested4DTensor(img_tensor, images.sizes, images.mask)

        detections = self._detector(
            images, [sample.caption for sample in batch.samples]
        )
        img_embeddings = self._vision(images)
        text_embeddings = self._text(batch)

        graphs = [
            Graph.new(
                batch.samples[idx].caption,
                text_embeddings[idx],
                detections[idx],
                self._same_entity_edge,
            )
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
        nodes = graph.nodes(False)  # (BN, D)

        offsets = self._regression_head(nodes)  # (BN, 4)
        padded_boxes = torch.logit(padded_boxes)  # (BN, 4)
        padded_boxes = padded_boxes + offsets  # (BN, 4)
        padded_boxes = torch.sigmoid(padded_boxes)  # (BN, 4)
        boxes = BBoxes(padded_boxes, images_size, BBoxFormat.CXCYWH, True)  # (BN, 4)

        nodes = graph.nodes(True)  # (B, N, D)
        nodes = self._proj(nodes)  # (B, N, D)
        graph = graph.new_like(nodes, graph.edges(True))

        sentences = torch.stack([txt.sentence for txt in text_embeddings])
        entity_mask = nn.utils.rnn.pad_sequence(
            [det.entities for det in detections],
            batch_first=True,
            padding_value=1,
        )  # (B, N)
        entity_mask = entity_mask != 0  # (B, N)

        return ModelOutput(
            sentences=sentences,
            graph=graph,
            boxes=boxes,
            mask=entity_mask,
            original_sizes=[
                tuple(sample.image.shape[1:])  # type: ignore
                for sample in batch.samples
            ],
        )
