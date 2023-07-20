##
##
##

import torch
from jaxtyping import Float
from torch import Tensor, nn
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from deepsight.data.structs import (
    Batch,
    BoundingBoxes,
    BoundingBoxFormat,
    ODInput,
    ODOutput,
)


class OwlViT(nn.Module):
    """Wrapper around the OwlViT model for open-set detection.

    With respect to the original OwlViT model, this wrapper adds the possibility
    to return the bounding boxes even when the confidence of the entity is below
    a certain threshold.
    """

    def __init__(self, threshold: float, num_boxes: int | None = None) -> None:
        """Initializes the OwlViT model.

        Parameters
        ----------
        threshold : float
            The threshold used to determine whether an entity is present in the
            input image.
        num_boxes : int | None
            If not None, when an entity is not found with a confidence above the
            `threshold`, the model will return the top `num_boxes` boxes with the
            highest confidence scores for that entity. If None, the model will
            return only the entities that are found with a confidence above the
            `threshold`. Defaults to None.
        """

        super().__init__()

        self._threshold = threshold
        self._num_boxes = num_boxes

        self._dummy = nn.Parameter(torch.empty(0))

        model_id = "google/owlvit-base-patch32"
        processor = OwlViTProcessor.from_pretrained(model_id)
        model = OwlViTForObjectDetection.from_pretrained(model_id)

        self.processor = processor
        self.owlvit = model.owlvit
        self.class_head = model.class_head
        self.box_head = model.box_head

        self.layer_norm = model.layer_norm

    def _get_boxes(
        self, image_embeds: Float[Tensor, "B L D"]
    ) -> Float[Tensor, "B L 4"]:
        L = image_embeds.shape[1]
        side = int(L**0.5)
        device = image_embeds.device
        dtype = image_embeds.dtype

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(1, side + 1, device=device, dtype=dtype),
                torch.arange(1, side + 1, device=device, dtype=dtype),
                indexing="xy",
            ),
            dim=-1,
        )
        coords = coords / side
        coords = coords.view(L, 2)

        coords = torch.clamp(coords, 0.0, 1.0)  # (L, 2)
        coord_bias = torch.log(coords + 1e-4) - torch.log1p(-coords + 1e-4)

        size = torch.full_like(coord_bias, 1.0 / side)  # (L, 2)
        size_bias = torch.log(size + 1e-4) - torch.log1p(-size + 1e-4)

        box_bias = torch.cat((coord_bias, size_bias), dim=-1)  # (L, 4)

        pred_boxes: Tensor = self.box_head(image_embeds)  # (B, L, 4)
        pred_boxes = pred_boxes + box_bias  # (B, L, 4)
        pred_boxes = torch.sigmoid(pred_boxes)  # (B, L, 4)

        return pred_boxes

    def forward(self, inputs: Batch[ODInput]) -> Batch[ODOutput]:
        images = [inp.image.to_pil().data for inp in inputs]

        # Create list of list of entities and remove duplicates
        entities: list[list[str]] = []
        str_to_idx: list[dict[str, list[int]]] = []
        for inp in inputs:
            sample_entities = []
            sample_str_to_idx = {}
            for ent_idx, ent in enumerate(inp.entities):
                ent = f"a photo of a {ent}"
                if ent not in sample_str_to_idx:
                    sample_str_to_idx[ent] = [ent_idx]
                    sample_entities.append(ent)
                else:
                    sample_str_to_idx[ent].append(ent_idx)

            entities.append(sample_entities)
            str_to_idx.append(sample_str_to_idx)

        B = len(images)
        max_queries = max(len(ent) for ent in entities)

        tmp = self.processor(
            images=images, text=entities, return_tensors="pt", truncation=True
        )
        pixel_values = tmp["pixel_values"].to(self._dummy.device)
        input_ids = tmp["input_ids"].to(self._dummy.device)
        attention_mask = tmp["attention_mask"].to(self._dummy.device)

        outputs = self.owlvit(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        image_tokens = outputs.vision_model_output[0]  # (B, 1+L, D)
        image_tokens = self.owlvit.vision_model.post_layernorm(
            image_tokens
        )  # (B, 1+L, D)

        class_token = image_tokens[:, :1, :]  # (B, 1, D)
        image_patches = image_tokens[:, 1:, :]  # (B, L, D)
        image_embeds = image_patches * class_token  # (B, L, D)
        image_embeds = self.layer_norm(image_embeds)  # (B, L, D)

        query_embeds = outputs[-4]  # (BQ, D) where BQ = B * max_queries
        query_embeds = query_embeds.view(B, max_queries, -1)  # (B, Q, D)

        query_mask = torch.zeros(
            (B, max_queries),
            dtype=torch.bool,
            device=query_embeds.device,
        )
        for sample_idx, sample_entities in enumerate(entities):
            query_mask[sample_idx, : len(sample_entities)] = True

        # (B, L, Q) means that each image patch is compared to each query
        # embedding, and the result is a scalar value representing the
        # similarity between the two.
        pred_logits, _ = self.class_head(image_embeds, query_embeds, query_mask)
        pred_boxes = self._get_boxes(image_embeds)

        probs, labels = pred_logits.max(dim=-1)  # (B, L)
        if self._num_boxes is not None:
            _, top_index_per_query = torch.topk(
                pred_logits, self._num_boxes, dim=1
            )  # (B, K, Q)

        scores = probs.sigmoid()  # (B, L)

        results = []
        for sample_idx in range(B):
            mask = scores[sample_idx] > self._threshold

            sample_pred_boxes = pred_boxes[sample_idx, mask]  # (N, 4)
            sample_pred_labels = labels[sample_idx, mask]  # (N,)
            sample_pred_scores = scores[sample_idx, mask]  # (N,)

            boxes_list: list[Tensor] = []
            labels_list: list[int] = []
            scores_list: list[Tensor] = []

            for ent_idx, ent in enumerate(entities[sample_idx]):
                indices = sample_pred_labels == ent_idx
                num_found = indices.sum().item()
                if num_found > 0:
                    # entity found in image
                    # add all duplicates to the list
                    for j in str_to_idx[sample_idx][ent]:
                        boxes_list.append(sample_pred_boxes[indices])
                        labels_list.extend([j] * num_found)
                        scores_list.append(sample_pred_scores[indices])
                elif self._num_boxes is not None:
                    # entity not found in image
                    # add top K boxes for entity
                    topk_boxes = pred_boxes[
                        sample_idx, top_index_per_query[sample_idx, :, ent_idx]
                    ]

                    topk_scores = scores[
                        sample_idx, top_index_per_query[sample_idx, :, ent_idx]
                    ]

                    for j in str_to_idx[sample_idx][ent]:
                        boxes_list.append(topk_boxes)
                        labels_list.extend([j] * self._num_boxes)
                        scores_list.append(topk_scores)

            sample_labels = torch.tensor(
                labels_list, dtype=torch.long, device=self._dummy.device
            )

            boxes = BoundingBoxes(
                tensor=torch.cat(boxes_list, dim=0),
                images_size=inputs[sample_idx].image.size,
                format=BoundingBoxFormat.CXCYWH,
                normalized=True,
            )

            sample_scores = torch.cat(scores_list, dim=0)

            results.append(
                ODOutput(
                    boxes=boxes,
                    entities=sample_labels,
                    scores=sample_scores,
                )
            )

        return Batch(results)

    def __call__(self, inputs: Batch[ODInput]) -> Batch[ODOutput]:
        """Given a batch of images and entities to be detected, returns a list
        of OSDOutput objects containing the bounding boxes of the detected entities.

        .. note::
            In case of duplicate entities for the same image, this implementation will
            return the same bounding boxes for both entities. This is different from the
            implementation of HuggingFace's OwlViT model which returns the bounding
            boxes only for one of the entities (usually the first one).

        Parameters
        ----------
        inputs : Batch[OSDInput]
            A Batch object containing OSDInput objects containing the images and
            entities.

        Returns
        -------
        Batch[OSDOutput]
            A Batch object containing the OSDOutput objects containing the bounding
            boxes.
        """

        return super().__call__(inputs)  # type: ignore
