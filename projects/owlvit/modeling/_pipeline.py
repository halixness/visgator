##
##
##

from typing_extensions import Self

from deepsight.data.structs import Batch, RECInput, RECOutput
from deepsight.modeling.pipeline import Pipeline as _Pipeline

from ._config import Config
from ._criterion import Criterion
from ._model import Model
from ._postprocessor import PostProcessor
from ._preprocessor import PreProcessor


class Pipeline(_Pipeline[RECInput, RECOutput, Batch[RECInput], Batch[RECOutput]]):
    @classmethod
    def new_for_rec(cls, config: Config) -> Self:  # type: ignore
        preprocessor = PreProcessor()
        model = Model(config)
        criterion = Criterion()
        postprocessor = PostProcessor()

        return cls(
            name="OwlViT",
            preprocessor=preprocessor,
            model=model,
            postprocessor=postprocessor,
            criterion=criterion,
        )
