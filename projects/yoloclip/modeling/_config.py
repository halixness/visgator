##
##
##

from dataclasses import dataclass
from typing import Any

import serde
from typing_extensions import Self

from deepsight.modeling.detectors import YOLOModel
from deepsight.modeling.layers.clip import Models
from deepsight.modeling.pipeline import Config as _Config


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True, slots=True)
class Config(_Config):
    clip: Models = Models.ViT_B_32_224
    yolo: YOLOModel = YOLOModel.MEDIUM
    box_threshold: float = 0.25

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return serde.from_dict(cls, d)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)
