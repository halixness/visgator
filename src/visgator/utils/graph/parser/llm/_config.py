##
##
##

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import serde
from typing_extensions import Self

from visgator.utils.graph.parser import Config as _Config


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True, slots=True)
class Config(_Config):
    """Configuration for the LLM parser."""

    prompt: Path
    model: str
    batch_size: int = 1

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)
