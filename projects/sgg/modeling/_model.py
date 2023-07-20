##
##
##

import torch
from jaxtyping import Integer
from torch import Tensor, nn

from deepsight.data.structs import (
    Batch,
    BoundingBoxes,
    BoundingBoxFormat,
    ODInput,
    ODOutput,
    SceneGraph,
)
from deepsight.modeling.detectors import OwlViT
from deepsight.modeling.layers import clip
from deepsight.modeling.pipeline import Model as _Model
from deepsight.utils.torch import BatchedGraphs, Graph

from ._config import Config
from ._decoder import Decoder
from ._structs import ModelInput, ModelOutput, TextEmbeddings


class Model(_Model[ModelInput, ModelOutput]):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.vision_encoder = clip.VisionEncoder(
            config.encoders.model,
            config.encoders.output_dim,
        )

        self.text_encoder = clip.TextEncoder(
            config.encoders.model,
            config.encoders.output_dim,
        )

        self.detector = OwlViT(
            config.detector.box_threshold, config.detector.num_queries
        )

        self.same_entity_edge = nn.Parameter(torch.randn(1, config.encoders.output_dim))
        self.decoder = Decoder(config.decoder)

        self.projection = nn.Linear(
            config.decoder.hidden_dim, config.decoder.hidden_dim
        )
        self.regression_head = nn.Sequential(
            nn.Linear(config.decoder.hidden_dim, config.decoder.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.decoder.dropout),
            nn.Linear(config.decoder.hidden_dim, 4),
        )

    def _get_text_embeddings(self, inputs: ModelInput) -> list[TextEmbeddings]:
        texts = []
        for caption, graph in zip(inputs.captions, inputs.graphs):
            texts.append(caption)
            texts.extend(f"a photo of {e.phrase}" for e in graph.entities())
            texts.extend(
                f"a photo of {r.subject.phrase} {r.relation} {r.object.phrase}"
                for r in graph.triplets(None, False, False)
            )

        tmp = self.text_encoder(texts)

        embeddings = []
        count = 0

        for graph in inputs.graphs:
            caption_emb = tmp[count]
            count += 1

            entities_emb = tmp[count : count + len(graph.entities())]
            count += len(graph.entities())

            num_relations = len(graph.triplets(None, True, True))
            relations_emb = tmp[count : count + num_relations]
            count += num_relations

            embeddings.append(
                TextEmbeddings(
                    entities=entities_emb,
                    relations=relations_emb,
                    caption=caption_emb,
                )
            )

        return embeddings

    def _get_detections(self, inputs: ModelInput) -> list[ODOutput]:
        batch = Batch(
            [
                ODInput(image, [e.noun for e in graph.entities()])
                for image, graph in zip(inputs.images, inputs.graphs)
            ]
        )

        return list(self.detector(batch))

    def _get_graph(
        self,
        graph: SceneGraph,
        embeddings: TextEmbeddings,
        detections: ODOutput,
    ) -> Graph:
        device = embeddings.entities.device
        num_relations = embeddings.relations.shape[0]

        edge_index_list: list[Integer[Tensor, "2 N"]] = []
        rel_index_list: list[int] = []

        for det_idx, detection in enumerate(detections.entities):
            entity_idx = int(detection)

            # add relations between entities based on the scene graph
            for rel in graph.triplets(entity_idx, True, True):
                end = (detections.entities == rel.object).nonzero(as_tuple=True)[0]
                end = end[None]  # (1, K)

                start = torch.tensor([det_idx], device=device).expand_as(end)
                indexes = torch.cat([start, end], dim=0)  # (2, K)

                edge_index_list.append(indexes)
                rel_index_list.extend([rel.relation] * indexes.shape[1])

            # add relations between instances of the same entity
            end = (detections.entities == entity_idx).nonzero(as_tuple=True)[0]
            end = end[None]  # (1, K)

            start = torch.tensor([det_idx], device=device).expand_as(end)
            indexes = torch.cat([start, end], dim=0)  # (2, K)

            edge_index_list.append(indexes)
            rel_index_list.extend([num_relations] * indexes.shape[1])

        edge_indices = torch.cat(edge_index_list, dim=1)  # (2, E)

        relations_emb = torch.cat([embeddings.relations, self.same_entity_edge])
        rel_indices = torch.tensor(rel_index_list, device=device)
        relations = relations_emb[rel_indices]  # (E, D)

        nodes = embeddings.entities[detections.entities]  # (N, D)

        return Graph(
            nodes=nodes,
            edges=relations,
            edge_indices=edge_indices,
        )

    def forward(self, inputs: ModelInput) -> ModelOutput:
        features = self.vision_encoder(inputs.features)
        embeddings = self._get_text_embeddings(inputs)
        detections = self._get_detections(inputs)

        tmp = [
            self._get_graph(graph, embedding, detection)
            for graph, embedding, detection in zip(
                inputs.graphs, embeddings, detections
            )
        ]

        # Build decoder inputs
        graph = BatchedGraphs.from_list(tmp)
        boxes = BoundingBoxes.cat([detection.boxes for detection in detections])
        graphs = self.decoder(features, graph, boxes)

        # Compute new boxes
        base_boxes = BoundingBoxes.pad_sequence(
            [detection.boxes for detection in detections]
        )  # (B, N, 4)
        base_boxes = base_boxes.to_cxcywh().normalize()

        new_boxes = []
        for idx in range(len(graphs)):
            graph = graphs[idx]

            nodes = graph.nodes(pad_value=0)  # (B, N, D)
            offsets = self.regression_head(nodes)  # (B, N, 4)
            box_tensor = torch.logit(base_boxes.tensor) + offsets
            box_tensor = torch.sigmoid(box_tensor)
            box = BoundingBoxes(
                box_tensor,
                base_boxes.images_size,
                format=BoundingBoxFormat.CXCYWH,
                normalized=True,
            )
            new_boxes.append(box)

            nodes = graph.nodes(None)  # (N, D)
            nodes = self.projection(nodes)  # (N, D)
            graphs[idx] = graph.new_like(nodes=nodes, clone=False)

        max_detections = max(len(detection.entities) for detection in detections)
        padded_entities = torch.nn.utils.rnn.pad_sequence(
            [detection.entities for detection in detections],
            batch_first=True,
            padding_value=max_detections,
        )

        return ModelOutput(
            captions=torch.stack([embedding.caption for embedding in embeddings]),
            graphs=graphs,
            boxes=new_boxes,
            padded_entities=padded_entities,
        )
