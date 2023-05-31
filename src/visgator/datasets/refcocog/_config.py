##
##
##

"""Configurations for RefCOCO dataset."""

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import serde
from typing_extensions import Self

from .._config import Config as _Config


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
class Config(_Config):
    """Configuration for RefCOCOg dataset."""

    path: Path
    split_provider: SplitProvider

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)
