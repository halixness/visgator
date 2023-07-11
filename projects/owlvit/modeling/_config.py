##
##
##

from dataclasses import dataclass
from typing import Any

import serde
from typing_extensions import Self

from deepsight.modeling.pipeline import Config as _Config


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True, slots=True)
class Config(_Config):
    box_threshold: float = 0.2

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {}
