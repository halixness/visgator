##
##
##

"""Configurations for RefCOCO dataset."""

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import serde
from typing_extensions import Self

from visgator.datasets import Config as _Config
from visgator.utils.graph import SceneGraphParserType


class SplitProvider(enum.Enum):
    GOOGLE = "google"
    UMD = "umd"

    @classmethod
    def from_str(cls, split: str) -> Self:
        return cls[split.upper().strip()]

    def __str__(self) -> str:
        return self.value


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for RefCOCOg dataset generation."""

    num_workers: Optional[int] = None
    chunksize: int = 128
    parser: SceneGraphParserType = SceneGraphParserType.SPACY

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class Config(_Config):
    """Configuration for RefCOCOg dataset."""

    path: Path
    split_provider: SplitProvider
    generation: GenerationConfig = serde.field(
        default=GenerationConfig(),
        serializer=GenerationConfig.to_dict,
        deserializer=GenerationConfig.from_dict,
    )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)
