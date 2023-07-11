##
##
##

from deepsight.data.structs import Batch, RECInput, RECOutput
from deepsight.modeling.pipeline import PreProcessor as _PreProcessor


class PreProcessor(_PreProcessor[RECInput, RECOutput, Batch[RECInput]]):
    def forward(
        self,
        inputs: Batch[RECInput],
        targets: Batch[RECOutput] | None,
    ) -> Batch[RECInput]:
        return inputs
