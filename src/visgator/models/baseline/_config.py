##
##
##

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import serde
from typing_extensions import Self

from visgator.models import Config as _Config


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class Config(_Config):
    """Configuration for baseline model."""

    model: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)
