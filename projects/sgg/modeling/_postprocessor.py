##
##
##

import torch

from deepsight.data.structs import Batch, RECOutput
from deepsight.modeling.pipeline import PostProcessor as _PostProcessor

from ._structs import ModelOutput


class PostProcessor(_PostProcessor[ModelOutput, RECOutput]):
    def forward(self, output: ModelOutput) -> Batch[RECOutput]:
        B, N = output.padded_entities.shape
        subject_mask = output.padded_entities != 0  # (B, N)

        queries = output.captions.unsqueeze(1)  # (B, 1, D)
        keys = output.graphs[-1].nodes(pad_value=0.0)  # (B, N, D)

        similarity = torch.cosine_similarity(queries, keys, dim=-1)  # (B, N)
        similarity.masked_fill_(subject_mask, -torch.inf)  # (B, N)

        idx = similarity.max(dim=1)[1]  # (B,)

        boxes = output.boxes[-1][torch.arange(B), idx]  # (B, 4)

        return Batch([RECOutput(box=boxes[i]) for i in range(B)])
