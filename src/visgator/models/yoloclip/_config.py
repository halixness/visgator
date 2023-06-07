##
##
##

from __future__ import annotations

import enum
from dataclasses import dataclass

import serde
from typing_extensions import Self

from visgator.models import Config as _Config


class YOLOModel(enum.Enum):
    NANO = "n"
    SMALL = "s"
    MEDIUM = "m"
    LARGE = "l"
    EXTRA = "x"

    def weights(self) -> str:
        return f"yolov8{self.value}.pt"

    @classmethod
    def from_str(cls, s: str) -> Self:
        return cls[s.upper().strip()]


class CLIPModel(enum.Enum):
    ViT_B_32_224 = "B32"
    ViT_B_16_224 = "B16"
    ViT_L_14_224 = "L14"
    Vit_L_14_336 = "L14_336"

    @classmethod
    def from_str(cls, s: str) -> CLIPModel:
        s = s.strip()
        match s:
            case "B32":
                return cls.ViT_B_32_224
            case "B16":
                return cls.ViT_B_16_224
            case "L14":
                return cls.ViT_L_14_224
            case "L14_336":
                return cls.Vit_L_14_336
            case _:
                raise ValueError(f"Invalid CLIP model: {s}")

    def weights(self) -> str:
        match self:
            case self.ViT_B_32_224:
                return "openai/clip-vit-base-patch32"
            case self.ViT_B_16_224:
                return "openai/clip-vit-base-patch16"
            case self.ViT_L_14_224:
                return "openai/clip-vit-large-patch32"
            case self.Vit_L_14_336:
                return "openai/clip-vit-large-patch32-336"
            case _:
                raise ValueError(f"Invalid CLIP model: {self}")


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class Config(_Config):
    yolo: YOLOModel = serde.field(default=YOLOModel.MEDIUM)
    clip: CLIPModel = serde.field(default=CLIPModel.ViT_B_32_224)

    @classmethod
    def from_dict(cls, cfg: dict[str, str]) -> Self:
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, str]:
        return serde.to_dict(self)
