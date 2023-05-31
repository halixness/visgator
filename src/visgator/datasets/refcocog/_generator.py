##
##
##

import json

from typing_extensions import Self

from visgator.datasets import Generator as _Generator
from visgator.datasets import Split
from visgator.utils.batch import Caption
from visgator.utils.graph import SpacySceneGraphParser

from ._config import Config
from ._misc import get_preprocessed_samples


class Generator(_Generator):
    def __init__(self, config: Config) -> None:
        self._config = config

    @classmethod
    def from_config(cls, config: Config) -> Self:  # type: ignore
        return cls(config)

    def generate(self) -> None:
        split_samples = get_preprocessed_samples(
            self._config,
            [Split.TRAIN, Split.VALIDATION, Split.TEST],
        )

        parser = SpacySceneGraphParser()

        output: dict[str, list] = {}  # type: ignore
        for split, samples in split_samples.items():
            split_output: list[dict] = []  # type: ignore
            output[str(split)] = split_output

            for sample in samples:
                graph = parser.parse(sample.caption.sentence)
                new_sample = {
                    "image": sample.path.name,
                    "caption": Caption(sample.caption.sentence, graph).to_dict(),
                    "bbox": sample.bbox,
                }
                split_output.append(new_sample)
                break
            break

        output_path = (
            self._config.path / f"annotations/info_{self._config.split_provider}.json"
        )
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
