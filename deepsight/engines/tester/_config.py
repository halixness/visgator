##
##
##

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import serde
from typing_extensions import Self

from deepsight.data.dataset import Config as DatasetConfig
from deepsight.modeling.pipeline import Config as PipelineConfig
from deepsight.utils import Config as _Config
from deepsight.utils.torch import FloatType
from deepsight.utils.wandb import Config as WandbConfig


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True, slots=True)
class Config(_Config):
    dataset: DatasetConfig = serde.field(
        serializer=lambda x: x.to_dict(),
        deserializer=DatasetConfig.from_dict,
    )
    pipeline: PipelineConfig = serde.field(
        serializer=lambda x: x.to_dict(),
        deserializer=PipelineConfig.from_dict,
    )

    dir: Path = serde.field(
        default=Path("output"),
        serializer=lambda x: str(x),
    )

    wandb: WandbConfig = serde.field(
        default=WandbConfig(job_type="test"),
        serializer=lambda x: x.to_dict(),
        deserializer=WandbConfig.from_dict,
    )

    seed: int = 3407
    debug: bool = False
    device: str | None = None
    dtype: FloatType = FloatType.FLOAT32
    weights: Path | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return serde.from_dict(cls, d)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)
