##
##
##

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import serde
from typing_extensions import Self

from deepsight.modeling.layers.clip import Models
from deepsight.modeling.pipeline import Config as _Config


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True, slots=True)
class PreprocessorConfig:
    file: Path | None = serde.field(
        serializer=lambda x: str(x) if x is not None else None
    )
    token: str = serde.field(skip=True)
    side: int = 800
    max_side: int = 1333
    mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True, slots=True)
class EncodersConfig:
    output_dim: int = serde.field(skip=True)
    model: Models = Models.ViT_B_32_224


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True, slots=True)
class DetectorConfig:
    box_threshold: float = 0.25
    num_queries: int = 4


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True, slots=True)
class DecoderConfig:
    hidden_dim: int = serde.field(skip=True)
    num_layers: int = 3
    num_heads: int = 8
    epsilon_layer_scale: float = 0.1
    dropout: float = 0.1


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True, slots=True)
class CriterionConfig:
    l1_cost: float = 5.0
    giou_cost: float = 2.0
    similarity_cost: float = 2.0

    l1_weight: float = 5.0
    giou_weight: float = 2.0
    infonce_weight: float = 1.0
    temperature: float = 0.1

    auxiliary: bool = True
    num_layers: int = 3


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True, slots=True)
class Config(_Config):
    preprocessor: PreprocessorConfig
    encoders: EncodersConfig
    detector: DetectorConfig
    decoder: DecoderConfig
    criterion: CriterionConfig

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        dim = cfg.get("dim")
        if dim is None:
            raise ValueError("Model dimension is required.")

        encoders = cfg.setdefault("encoders", {})
        decoder = cfg.setdefault("decoder", {})

        encoders["output_dim"] = dim
        decoder["hidden_dim"] = dim

        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)
