##
##
##

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from visgator.models import Criterion as _Criterion
from visgator.utils.bbox import BBoxes, ops

from .._criterion import LossInfo
from ._config import CriterionConfig
from ._misc import ModelOutput


class Criterion(_Criterion[ModelOutput]):
    def __init__(self, config: CriterionConfig) -> None:
        super().__init__()

        self._l1_weight = config.l1_weight
        self._giou_weight = config.giou_weight
        self._info_nce_weight = config.info_nce_weight

        self._temperature = config.temperature

    @property
    def losses(self) -> list[LossInfo]:
        return [
            LossInfo("L1", self._l1_weight),
            LossInfo("GIoU", self._giou_weight),
            LossInfo("InfoNCE", self._info_nce_weight),
        ]

    # code taken from https://github.com/RElbers/info-nce-pytorch
    def _info_nce_loss(
        self,
        query: Float[Tensor, "B D"],
        pos_keys: Float[Tensor, "B D"],
        neg_keys: Float[Tensor, "M D"],
    ) -> Float[Tensor, ""]:
        B = query.shape[0]

        query = F.normalize(query, dim=-1)  # (B, D)
        pos_keys = F.normalize(pos_keys, dim=-1)  # (B, D)
        neg_keys = F.normalize(neg_keys, dim=-1)  # (M, D)

        pos_logits = torch.sum(query * pos_keys, dim=-1, keepdim=True)  # (B, 1)
        neg_logits = query @ neg_keys.T  # (B, M)

        logits = torch.cat((pos_logits, neg_logits), dim=-1)  # (B, 1+M)
        logits /= self._temperature  # (B, 1+M)

        labels = torch.zeros(B, dtype=torch.long, device=logits.device)  # (B,)
        info_nce_loss = F.cross_entropy(logits, labels, reduction="mean")

        return info_nce_loss

    def forward(
        self, output: ModelOutput, target: BBoxes
    ) -> dict[str, Float[Tensor, ""]]:
        B, N = output.mask.shape

        tgt_boxes = target.to_xyxy().normalize().tensor
        out_boxes = output.boxes.to_xyxy().normalize().tensor

        tgt_boxes = tgt_boxes.unsqueeze(1)  # (B, 1, 4)
        tgt_boxes = tgt_boxes.expand(-1, N, -1)  # (B, N, 4)
        tgt_boxes = tgt_boxes.flatten(0, 1)  # (BN, 4)

        l1_loss = torch.cdist(out_boxes, tgt_boxes, p=1).diagonal()  # (BN,)
        giou = ops.generalized_box_iou_pairwise(out_boxes, tgt_boxes)  # (BN,)
        giou_loss = -giou  # (BN,)

        l1_loss = l1_loss.view(B, N)  # (B, N)
        giou_loss = giou_loss.view(B, N)  # (B, N)

        matching_loss = (
            self._l1_weight * l1_loss + self._giou_weight * giou_loss
        )  # (B, N)
        matching_loss.masked_fill_(output.mask, torch.inf)  # (B, N)
        idx = torch.min(matching_loss, dim=1)[1]  # (B,)

        mask = output.mask.clone()  # (B, N)
        mask[torch.arange(B), idx] = True  # (B, N)

        nodes = output.graph.nodes(True)  # (B, N, D)

        query = output.sentences  # (B, D)
        pos_keys = nodes[torch.arange(B), idx]  # (B, D)
        neg_keys = nodes[~mask]  # (M, D)
        info_nce_loss = self._info_nce_loss(query, pos_keys, neg_keys)

        l1_loss = l1_loss[torch.arange(B), idx].mean()
        giou_loss = giou_loss[torch.arange(B), idx].mean()

        return {
            "L1": l1_loss,
            "GIoU": giou_loss,
            "InfoNCE": info_nce_loss,
        }
