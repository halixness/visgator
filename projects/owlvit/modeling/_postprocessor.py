##
##
##

from deepsight.data.structs import Batch, RECOutput
from deepsight.modeling.pipeline import PostProcessor as _PostProcessor


class PostProcessor(_PostProcessor[Batch[RECOutput], RECOutput]):
    def forward(self, output: Batch[RECOutput]) -> Batch[RECOutput]:
        return output
