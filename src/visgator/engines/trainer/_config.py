##
##
##

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import serde
from typing_extensions import Self

from visgator.datasets import Config as DatasetConfig
from visgator.lr_schedulers import Config as LRSchedulerConfig
from visgator.models import Config as ModelConfig
from visgator.optimizers import Config as OptimizerConfig

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
    save: bool = False

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
        if self.args is None:
            return {"enabled": self.enabled}
        else:
            return {"enabled": self.enabled, **self.args.to_dict()}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Configuration for training."""

    dir: Path
    wandb: WandbConfig
    params: dict[str, Any]
    debug: bool = False

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        """Deserializes a dictionary into a Config object."""
        wandb = WandbConfig.from_dict(cfg.pop("wandb", {}))
        dir = Path(cfg.pop("dir", "output"))
        debug = bool(cfg.pop("debug", False))

        return cls(dir=dir, wandb=wandb, params=cfg, debug=debug)

    def to_dict(self) -> dict[str, Any]:
        """Serializes a Config object into a dictionary."""
        return {
            "dir": str(self.dir),
            "wandb": self.wandb.to_dict(),
            "debug": self.debug,
        }


# ---------------------------------------------------------------------------
# Training Parameters
# ---------------------------------------------------------------------------


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class Params:
    """Parameters for training."""

    num_epochs: int
    train_batch_size: int
    eval_batch_size: int
    dataset: DatasetConfig = serde.field(
        serializer=DatasetConfig.to_dict,
        deserializer=DatasetConfig.from_dict,
    )
    model: ModelConfig = serde.field(
        serializer=ModelConfig.to_dict,
        deserializer=ModelConfig.from_dict,
    )
    optimizer: OptimizerConfig = serde.field(
        serializer=OptimizerConfig.to_dict,
        deserializer=OptimizerConfig.from_dict,
    )
    lr_scheduler: LRSchedulerConfig = serde.field(
        serializer=LRSchedulerConfig.to_dict,
        deserializer=LRSchedulerConfig.from_dict,
    )

    seed: int = 3407
    compile: bool = True
    checkpoint_interval: int = 1
    eval_interval: int = 1
    gradient_accumulation_steps: int = 1
    device: Optional[str] = None
    mixed_precision: bool = True
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

        if self.train_batch_size < 1:
            raise ValueError(
                f"Invalid batch size: {self.train_batch_size}. Must be greater than 0."
            )

        if self.train_batch_size % self.gradient_accumulation_steps != 0:
            raise ValueError(
                f"Invalid batch size: {self.train_batch_size}. Must be divisible by "
                f"gradient accumulation steps: {self.gradient_accumulation_steps}."
            )

        if self.eval_interval < 1 or self.eval_interval > self.num_epochs:
            raise ValueError(
                "Invalid eval interval: "
                f"{self.eval_interval}. Must be between 1 and {self.num_epochs}."
            )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        """Deserializes a dictionary into a Params object."""
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        """Serializes a Params object into a dictionary."""
        return serde.to_dict(self)
