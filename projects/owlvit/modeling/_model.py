##
##
##

from deepsight.data.structs import Batch, ODInput, RECInput, RECOutput
from deepsight.modeling.detectors import OwlViT
from deepsight.modeling.pipeline import Model as _Model

from ._config import Config


class Model(_Model[Batch[RECInput], Batch[RECOutput]]):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.detector = OwlViT(config.box_threshold, 1)

    def forward(self, inputs: Batch[RECInput]) -> Batch[RECOutput]:
        tmp = Batch([ODInput(inp.image, [inp.description]) for inp in inputs])
        results = self.detector(tmp)

        outputs = []
        for result in results:
            idx = result.scores.argmax()
            box = result.boxes[idx]
            outputs.append(RECOutput(box))

        return Batch(outputs)
