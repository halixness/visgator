##
##
##

import abc
import enum

from typing_extensions import Self

from visgator.utils.batch import Batch, BatchSample
from visgator.utils.bbox import BBox, BBoxes
from visgator.utils.factory import get_subclass

from ._config import Config


class Split(enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

    @classmethod
    def from_str(cls, value: str) -> Self:
        return cls[value.upper().strip()]

    def __str__(self) -> str:
        return self.value


class Dataset(abc.ABC):
    @classmethod
    def from_config(cls, config: Config, split: Split, debug: bool = False) -> Self:
        sub_cls = get_subclass(cls, config.name)
        return sub_cls.from_config(config, split, debug)

    @abc.abstractmethod
    def __getitem__(self, index: int) -> tuple[BatchSample, BBox]:
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @staticmethod
    def batchify(batch: list[tuple[BatchSample, BBox]]) -> tuple[Batch, BBoxes]:
        samples = tuple(sample for sample, _ in batch)
        bboxes = [bbox for _, bbox in batch]
        return Batch(samples), BBoxes.from_bboxes(bboxes)
