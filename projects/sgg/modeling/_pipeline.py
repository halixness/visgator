##
##
##

from typing_extensions import Self

from deepsight.data.structs import RECInput, RECOutput
from deepsight.modeling.pipeline import Pipeline as _Pipeline

from ._config import Config
from ._criterion import Criterion
from ._model import Model
from ._postprocessor import PostProcessor
from ._preprocessor import PreProcessor
from ._structs import ModelInput, ModelOutput


class Pipeline(_Pipeline[RECInput, RECOutput, ModelInput, ModelOutput]):
    @classmethod
    def new_for_rec(cls, config: Config) -> Self:  # type: ignore
        preprocessor = PreProcessor(config.preprocessor)
        model = Model(config)
        criterion = Criterion(config.criterion)
        postprocessor = PostProcessor()

        return cls(
            name="Scene Graph Grounder",
            preprocessor=preprocessor,
            model=model,
            criterion=criterion,
            postprocessor=postprocessor,
        )
