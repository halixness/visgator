##
##
##

import json
from pathlib import Path

from tqdm import tqdm
from typing_extensions import Self

from visgator.datasets import Generator as _Generator
from visgator.datasets import Split
from visgator.utils.batch import Caption
from visgator.utils.graph import SceneGraph
from visgator.utils.graph.parser import Parser

from ._config import Config
from ._misc import Sample, get_original_samples


class Generator(_Generator):
    def __init__(self, config: Config) -> None:
        if config.generation is None:
            raise ValueError("config.generation cannot be None.")

        self._config = config

    @classmethod
    def new(cls, config: Config) -> Self:  # type: ignore
        return cls(config)

    def _get_missing_intervals(
        self,
        dir: Path,
        start: int,
        end: int,
    ) -> list[tuple[int, int]]:
        files = dir.glob("*.json")
        # file format is <start>-<end>.json
        indexes = [tuple(map(int, f.stem.split("-"))) for f in files]
        indexes.sort(key=lambda i: i[0])

        current_start = start
        missing = []
        for s, e in indexes:
            if current_start < s:
                missing.append((current_start, s))
            elif current_start > s:
                raise ValueError("Overlapping intervals.")

            current_start = e

            if current_start >= end:
                break

        if current_start < end:
            missing.append((current_start, end))

        return missing

    def _merge(self, tmp: Path) -> None:
        files = tmp.glob("*.json")

        output = {}  # type: ignore
        for file in files:
            with file.open("r") as f:
                data = json.load(f)

            for key, value in data.items():
                output.setdefault(key, []).extend(value)

        output_file = (
            self._config.path / f"annotations/info_{self._config.split_provider}.json"
        )
        with output_file.open("w") as f:
            json.dump(output, f)

    def generate(self) -> None:
        assert self._config.generation is not None

        tmp_dir = self._config.path / f"annotations/tmp_{self._config.split_provider}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        samples = get_original_samples(self._config.path, self._config.split_provider)
        start = self._config.generation.start
        end = self._config.generation.end or len(samples)
        intervals = self._get_missing_intervals(tmp_dir, start, end)

        if len(intervals) == 0:
            self._merge(tmp_dir)
        else:
            samples.sort(key=lambda s: s[0].caption.sentence)
            self._parser = Parser.new(self._config.generation.parser)

            for start, end in intervals:
                batch = samples[start:end]
                sentences = [s[0].caption.sentence for s in batch]

                generator = tqdm(
                    zip(batch, self._parser.parse(sentences)),
                    total=len(batch),
                    desc=f"Generating {start}-{end}",
                )

                outputs = {}  # type: ignore
                sample: Sample
                split: Split
                graph: SceneGraph
                for (sample, split), graph in generator:
                    outputs.setdefault(str(split), []).append(
                        {
                            "image": sample.path.name,
                            "caption": Caption(
                                sample.caption.sentence, graph
                            ).to_dict(),
                            "bbox": sample.bbox,
                        }
                    )

                with open(tmp_dir / f"{start}-{end}.json", "w") as f:
                    json.dump(outputs, f)
