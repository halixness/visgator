##
##
##

import json
import multiprocessing

from tqdm.contrib.concurrent import process_map
from typing_extensions import Self

from visgator.datasets import Generator as _Generator
from visgator.datasets import Split
from visgator.utils.batch import Caption
from visgator.utils.graph import SceneGraphParser

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

        parser = SceneGraphParser.new(self._config.generation.parser)
        if self._config.generation.num_workers is not None:
            num_workers = self._config.generation.num_workers
        else:
            num_workers = multiprocessing.cpu_count() // 2

        output: dict[str, list] = {}  # type: ignore
        for split, samples in split_samples.items():
            split_output: list[dict] = []  # type: ignore
            output[str(split)] = split_output

            graphs = process_map(
                parser.parse,
                (sample.caption.sentence for sample in samples),
                max_workers=num_workers,
                chunksize=self._config.generation.chunksize,
                desc=f"Generating {split} split",
                total=len(samples),
            )

            output[str(split)] = [
                {
                    "image": sample.path.name,
                    "caption": Caption(sample.caption.sentence, graph).to_dict(),
                    "bbox": sample.bbox,
                }
                for sample, graph in zip(samples, graphs)
            ]

        output_path = (
            self._config.path / f"annotations/info_{self._config.split_provider}.json"
        )
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
