##
##
##

import torch

from deepsight.data.structs import Batch, BoundingBoxes, RECOutput
from deepsight.measures import Loss, Reduction
from deepsight.measures.losses import BoxL1Loss, GeneralizedBoxIoULoss, InfoNCELoss
from deepsight.modeling.pipeline import Criterion as _Criterion

from ._config import CriterionConfig
from ._structs import ModelOutput


class Criterion(_Criterion[ModelOutput, RECOutput]):
    def __init__(self, config: CriterionConfig) -> None:
        super().__init__()

        self.auxiliary = config.auxiliary
        self.num_layers = config.num_layers

        self.l1_cost = config.l1_cost
        self.giou_cost = config.giou_cost
        self.similarity_cost = config.similarity_cost

        self.l1_weight = config.l1_weight
        self.giou_weight = config.giou_weight
        self.infonce_weight = config.infonce_weight

        self.l1_loss = BoxL1Loss(reduction=Reduction.NONE)
        self.giou_loss = GeneralizedBoxIoULoss(reduction=Reduction.NONE)
        self.infonce_loss = InfoNCELoss(
            temperature=config.temperature, reduction=Reduction.MEAN
        )

    def losses_names(self) -> list[str]:
        losses = []
        if self.auxiliary:
            losses += [f"L1_{i}" for i in range(self.num_layers)]
            losses += [f"GIoU_{i}" for i in range(self.num_layers)]
            losses += [f"InfoNCE_{i}" for i in range(self.num_layers)]
        else:
            losses += ["L1", "GIoU", "InfoNCE"]

        return losses

    def _compute_layer_loss(
        self,
        output: ModelOutput,
        tgt_boxes: BoundingBoxes,
        layer_idx: int,
    ) -> list[Loss]:
        """Computes the loss for the output of a single layer.

        Parameters
        ----------
        output : ModelOutput
            The output of the model.
        tgt_boxes : BoundingBoxes
            The target boxes. The tensor has shape (B, N, 4).
        layer_idx : int
            The index of the layer.
        """

        B, N = output.padded_entities.shape
        subject_mask = output.padded_entities != 0
        padding_mask = output.padded_entities == N

        out_boxes = output.boxes[layer_idx].to_cxcywh().normalize()  # (B, N, 4)

        l1_loss = self.l1_loss(out_boxes, tgt_boxes)  # (B, N)
        giou_loss = self.giou_loss(out_boxes, tgt_boxes)  # (B, N)

        nodes = output.graphs[layer_idx].nodes(pad_value=0.0)  # (B, N, D)
        captions = output.captions.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)
        similarity = torch.cosine_similarity(nodes, captions, dim=-1)  # (B, N)

        cost = (
            self.l1_cost * l1_loss
            + self.giou_cost * giou_loss
            - self.similarity_cost * similarity
        )

        cost = cost.masked_fill_(subject_mask, torch.inf)  # (B, N)
        idx = cost.min(dim=1)[1]  # (B,)

        pos_mask = torch.zeros_like(output.padded_entities, dtype=torch.bool)  # (B, N)
        pos_mask[torch.arange(B), idx] = True

        nodes = output.graphs[layer_idx].nodes(pad_value=0.0)  # (B, N, D)
        queries = output.captions  # (B, D)
        pos_keys = nodes[pos_mask]
        neg_mask = torch.logical_xor(pos_mask, ~padding_mask)
        neg_keys = nodes[neg_mask]
        infonce_loss = self.infonce_loss(queries, pos_keys, neg_keys)  # (B,)

        l1_loss = l1_loss[pos_mask].mean()
        giou_loss = giou_loss[pos_mask].mean()

        if layer_idx == -1:
            return [
                Loss("L1", l1_loss, self.l1_weight),
                Loss("GIoU", giou_loss, self.giou_weight),
                Loss("InfoNCE", infonce_loss, self.infonce_weight),
            ]
        else:
            return [
                Loss(f"L1_{layer_idx}", l1_loss, self.l1_weight),
                Loss(f"GIoU_{layer_idx}", giou_loss, self.giou_weight),
                Loss(f"InfoNCE_{layer_idx}", infonce_loss, self.infonce_weight),
            ]

    def forward(self, output: ModelOutput, targets: Batch[RECOutput]) -> list[Loss]:
        B, N = output.padded_entities.shape

        tgt_boxes = BoundingBoxes.stack([tgt.box for tgt in targets], dim=0)  # (B, 4)
        tgt_boxes = tgt_boxes.to_cxcywh().normalize()  # (B, 4)
        tgt_boxes = tgt_boxes.unsqueeze(1).expand(-1, N, -1)  # (B, N, 4)

        if self.auxiliary:
            losses = []
            for i in range(self.num_layers):
                losses += self._compute_layer_loss(output, tgt_boxes, i)
        else:
            losses = self._compute_layer_loss(output, tgt_boxes, -1)

        return losses
