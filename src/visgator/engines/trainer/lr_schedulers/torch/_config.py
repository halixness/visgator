##
##
##

from dataclasses import dataclass, field
from typing import Any

import serde
from typing_extensions import Self

from .._config import Config as _Config


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class Config(_Config):
    """Configuration for PyTorch lr schedulers."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)
