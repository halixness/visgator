##
##
##

from __future__ import annotations

import enum
from typing import Any

from typing_extensions import Self

from .._config import Config as BaseConfig


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


class Config(BaseConfig):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)

        self._yolo = YOLOModel.from_str(cfg.get("yolo", "nano"))
        self._clip = CLIPModel.from_str(cfg.get("clip", "B32"))

    @property
    def yolo(self) -> YOLOModel:
        return self._yolo

    @property
    def clip(self) -> CLIPModel:
        return self._clip
