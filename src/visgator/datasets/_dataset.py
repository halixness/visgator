##
##
##

from __future__ import annotations

import abc
import enum

from visgator.utils.batch import Batch, BatchSample
from visgator.utils.bbox import BBox, BBoxes
from visgator.utils.misc import instantiate

from ._config import Config


class Split(enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

    def __str__(self) -> str:
        return self.value


class Dataset(abc.ABC):
    def __init__(self, config: Config, split: Split, debug: bool) -> None:
        super().__init__()

    @staticmethod
    def from_config(config: Config, split: Split, debug: bool = False) -> Dataset:
        child_module = config.name.lower()
        parent_module = ".".join(Dataset.__module__.split(".")[:-1])
        module = f"{parent_module}.{child_module}"
        class_path = f"{module}.Dataset"

        return instantiate(class_path, Dataset, config, split, debug)  # type: ignore

    @abc.abstractmethod
    def __getitem__(self, index: int) -> tuple[BatchSample, BBox]:
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @staticmethod
    def batchify(batch: list[tuple[BatchSample, BBox]]) -> tuple[Batch, BBoxes]:
        samples, bboxes = zip(*batch)
        return Batch(samples), BBoxes(bboxes)
