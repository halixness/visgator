##
##
##

import torch
import torch.nn.functional as F

from visgator.models import PostProcessor as _PostProcessor
from visgator.utils.bbox import BBoxes

from ._misc import ModelOutput


class PostProcessor(_PostProcessor[ModelOutput]):
    def forward(self, output: ModelOutput) -> BBoxes:
        query = output.sentences.unsqueeze(1)  # (B, 1, D)
        keys = output.graph.nodes(True)  # (B, N, D)
        B, N = output.mask.shape[:2]

        query = F.normalize(query, dim=-1)  # (B, 1, D)
        keys = F.normalize(keys, dim=-1)  # (B, N, D)

        logits = torch.sum(query * keys, dim=-1)  # (B, N)
        logits.masked_fill_(output.mask, -torch.inf)  # (B, N)

        idx = torch.max(logits, dim=1)[1]  # (B,)

        bboxes = output.boxes.tensor.view(B, N, 4)  # (B, N, 4)
        images_size = output.boxes.images_size.view(B, N, 2)  # (B, N, 2)
        bboxes = bboxes[torch.arange(B), idx]  # (B, 4)
        images_size = images_size[torch.arange(B), idx]  # (B, 2)

        if not output.boxes.normalized:
            raise RuntimeError("Expected normalized bounding boxes.")

        return BBoxes(bboxes, output.original_sizes, output.boxes.format, True)
