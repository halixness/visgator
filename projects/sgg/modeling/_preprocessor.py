##
##
##

import json
from typing import Any

from deepsight.data.structs import Batch, RECInput, RECOutput, SceneGraph
from deepsight.data.transformations import Compose, Resize, Standardize
from deepsight.modeling.pipeline import PreProcessor as _PreProcessor
from deepsight.utils.torch import Batched3DTensors

from ._config import PreprocessorConfig
from ._structs import ModelInput


class PreProcessor(_PreProcessor[RECInput, RECOutput, ModelInput]):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__()

        self._preparsed: dict[str, dict[str, Any]] = {}
        if config.file is not None:
            with config.file.open("r") as f:
                self._preparsed = json.load(f)

        # self._parser = gpt.SceneGraphParser(config.token)

        self._transform = Compose(
            [
                Resize(config.side, max_size=config.max_side, p=1.0),
                Standardize(config.mean, config.std, p=1.0),
            ],
            p=1.0,
        )

    def forward(
        self,
        inputs: Batch[RECInput],
        targets: Batch[RECOutput] | None,
    ) -> ModelInput:
        graphs = []
        for inp in inputs:
            if inp.description in self._preparsed:
                scene_graph = SceneGraph.from_dict(self._preparsed[inp.description])
                # remove all nodes not connected to the root node
                # since they will never pass messages to the root node
                # (directly or through other nodes)
                scene_graph = scene_graph.node_connected_component(0)
                graphs.append(scene_graph)
            else:
                raise NotImplementedError

        return ModelInput(
            images=[i.image for i in inputs],
            features=Batched3DTensors.from_list(
                [self._transform(inp.image)[0].to_tensor().data for inp in inputs]
            ),
            captions=[i.description for i in inputs],
            graphs=graphs,
        )
