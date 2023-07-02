##
##
##

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import serde
from typing_extensions import Self

from visgator.models import Config as _Config


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class EncodersConfig:
    """Configuration for encoders."""

    hidden_dim: int = serde.field(skip=True)
    model: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)


@dataclass(frozen=True)
class GroundingDINOConfig:
    weights: Path
    config: Path

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for object detector."""

    gdino: Optional[GroundingDINOConfig] = serde.field(
        default=None,
        serializer=GroundingDINOConfig.to_dict,
        deserializer=GroundingDINOConfig.from_dict,
    )
    owlvit: Optional[str] = serde.field(default=None)
    box_threshold: float = serde.field(default=0.35)
    text_threshold: float = serde.field(default=0.25)
    max_detections: int = serde.field(default=50)

    def __post_init__(self) -> None:
        if self.box_threshold < 0 or self.box_threshold > 1:
            raise ValueError("box_threshold must be between 0 and 1.")

        if self.text_threshold < 0 or self.text_threshold > 1:
            raise ValueError("text_threshold must be between 0 and 1.")

        if self.max_detections < 1:
            raise ValueError("max_detections must be positive.")

        if self.gdino is None and self.owlvit is None:
            raise ValueError("Either gdino or owlvit must be set.")

        if self.gdino is not None and self.owlvit is not None:
            raise ValueError("Only one of gdino or owlvit must be set.")

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class DecoderConfig:
    """Configuration for decoder."""

    num_layers: int
    hidden_dim: int = serde.field(skip=True)
    num_heads: int = 8
    epsilon_layer_scale: float = 0.1  # our decoder has less than 18 layers
    dropout: float = serde.field(default=0.1, skip=True)

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive.")

        if self.num_heads < 1:
            raise ValueError("num_heads must be positive.")

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be between 0 and 1.")

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class HeadConfig:
    """Configuration for head."""

    hidden_dim: int
    num_layers: int
    num_heads: int = 8
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive.")

        if self.num_heads < 1:
            raise ValueError("num_heads must be positive.")

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be between 0 and 1.")

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class CriterionConfig:
    """Configuration for criterion."""

    l1_weight: float = 1.0
    giou_weight: float = 1.0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class Config(_Config):
    """Configuration for SceneGraphGrounder model."""

    encoders: EncodersConfig = serde.field(
        serializer=EncodersConfig.to_dict,
        deserializer=EncodersConfig.from_dict,
    )
    detector: DetectorConfig = serde.field(
        serializer=DetectorConfig.to_dict,
        deserializer=DetectorConfig.from_dict,
    )
    decoder: DecoderConfig = serde.field(
        serializer=DecoderConfig.to_dict,
        deserializer=DecoderConfig.from_dict,
    )
    head: HeadConfig = serde.field(
        serializer=HeadConfig.to_dict,
        deserializer=HeadConfig.from_dict,
    )
    criterion: CriterionConfig = serde.field(
        serializer=CriterionConfig.to_dict,
        deserializer=CriterionConfig.from_dict,
        default=CriterionConfig(),
    )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        hidden_dim = cfg.get("hidden_dim", None)
        if hidden_dim is None:
            raise ValueError("hidden_dim must be provided.")

        encoders = cfg.setdefault("encoders", {})
        decoder = cfg.setdefault("decoder", {})
        head = cfg.setdefault("head", {})

        encoders["hidden_dim"] = hidden_dim
        decoder["hidden_dim"] = hidden_dim
        head["hidden_dim"] = hidden_dim

        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)
