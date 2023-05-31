##
##
##

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import serde
from typing_extensions import Self

from .._config import Config as _Config


class CLIPModel(enum.Enum):
    ViT_B_32_224 = enum.auto()
    ViT_B_16_224 = enum.auto()
    ViT_L_14_224 = enum.auto()
    Vit_L_14_336 = enum.auto()

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
    """Configuration for baseline model."""

    clip: CLIPModel = serde.field(default=CLIPModel.ViT_B_32_224)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)
