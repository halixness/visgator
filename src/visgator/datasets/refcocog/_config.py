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
from visgator.utils.graph.parser import Config as ParserConfig


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

    parser: ParserConfig = serde.field(
        serializer=ParserConfig.to_dict,
        deserializer=ParserConfig.from_dict,
    )
    start: int = 0
    end: Optional[int] = None

    def __post_init__(self) -> None:
        if self.end is not None and self.end <= self.start:
            raise ValueError("end must be greater than start.")

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
    generation: Optional[GenerationConfig] = serde.field(
        default=None,
        serializer=GenerationConfig.to_dict,
        deserializer=GenerationConfig.from_dict,
    )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)
