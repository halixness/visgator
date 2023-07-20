##
##
##

from deepsight.data.structs import Batch, RECOutput
from deepsight.measures import Loss
from deepsight.modeling.pipeline import Criterion as _Criterion


class Criterion(_Criterion[Batch[RECOutput], RECOutput]):
    def losses_names(self) -> list[str]:
        raise NotImplementedError

    def forward(
        self,
        inputs: Batch[RECOutput],
        targets: Batch[RECOutput],
    ) -> list[Loss]:
        raise NotImplementedError
