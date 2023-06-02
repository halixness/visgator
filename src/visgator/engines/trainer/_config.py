##
##
##

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import serde
from typing_extensions import Self

from visgator.datasets import Config as DatasetConfig
from visgator.models import Config as ModelConfig

from .lr_schedulers import Config as LRSchedulerConfig
from .optimizers import Config as OptimizerConfig

# ---------------------------------------------------------------------------
# Wandb
# ---------------------------------------------------------------------------


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class WandbRunArgs:
    """Arguments for wandb run."""

    project: Optional[str] = "visual grounding"
    job_type: Optional[str] = "train"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[list[str]] = None
    notes: Optional[str] = None
    id: Optional[str] = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        """Deserializes a dictionary into a WandbRunArgs object."""
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        """Serializes a WandbRunArgs object into a dictionary."""
        return serde.to_dict(self)


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class WandbConfig:
    """Configuration for wandb."""

    args: Optional[WandbRunArgs] = serde.field(
        serializer=WandbRunArgs.to_dict,
        deserializer=WandbRunArgs.from_dict,
        metadata={"serde_flatten": True},
    )
    enabled: bool = True

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        """Deserializes a dictionary into a WandbConfig object."""
        enabled = bool(cfg.get("enabled", True))
        if not enabled:
            args = None
        else:
            args = WandbRunArgs.from_dict(cfg)

        return cls(args=args, enabled=enabled)

    def to_dict(self) -> dict[str, Any]:
        """Serializes a WandbConfig object into a dictionary."""
        return serde.to_dict(self)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class Config:
    """Configuration for training."""

    dir: Path
    wandb: WandbConfig = serde.field(
        serializer=WandbConfig.to_dict,
        deserializer=WandbConfig.from_dict,
    )
    params: dict[str, Any] = serde.field(skip=True, metadata={"serde_flatten": True})
    debug: bool = False

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        """Deserializes a dictionary into a Config object."""
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        """Serializes a Config object into a dictionary."""
        return serde.to_dict(self)


# ---------------------------------------------------------------------------
# Training Parameters
# ---------------------------------------------------------------------------


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class Params:
    """Parameters for training."""

    num_epochs: int
    batch_size: int
    mixed_precision: bool
    dataset: DatasetConfig = serde.field(deserializer=DatasetConfig.from_dict)
    model: ModelConfig = serde.field(deserializer=ModelConfig.from_dict)
    optimizer: OptimizerConfig = serde.field(deserializer=OptimizerConfig.from_dict)
    lr_scheduler: LRSchedulerConfig = serde.field(
        deserializer=LRSchedulerConfig.from_dict
    )

    seed: int = 3407
    compile: bool = True
    output_dir: Path = Path("output")
    checkpoint_interval: int = 1
    gradient_accumulation_steps: int = 1
    device: Optional[str] = None
    max_grad_norm: Optional[float] = None

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                "Invalid gradient accumulation steps: "
                f"{self.gradient_accumulation_steps}. Must be greater than 0."
            )

        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError(
                f"Invalid max gradient norm: {self.max_grad_norm}. "
                "Must be greater than 0."
            )

        if self.checkpoint_interval < 1:
            raise ValueError(
                "Invalid checkpoint interval: "
                f"{self.checkpoint_interval}. Must be greater than 0."
            )

        if self.batch_size < 1:
            raise ValueError(
                f"Invalid batch size: {self.batch_size}. Must be greater than 0."
            )

        if self.batch_size % self.gradient_accumulation_steps != 0:
            raise ValueError(
                f"Invalid batch size: {self.batch_size}. Must be divisible by "
                f"gradient accumulation steps: {self.gradient_accumulation_steps}."
            )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        """Deserializes a dictionary into a Params object."""
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        """Serializes a Params object into a dictionary."""
        return serde.to_dict(self)
