##
##
##

from dataclasses import dataclass
from typing import Any, Union

import serde
from typing_extensions import Self

from visgator.lr_schedulers import Config as _Config


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class Config(_Config):
    """Configuration for PyTorch lr schedulers."""

    name: str
    args: dict[str, Union[str, bool, int, float]] = serde.field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)
