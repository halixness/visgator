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


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class Config:
    """Configuration for training."""

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
    debug: bool = False
    compile: bool = True
    output_dir: Path = Path("output")
    checkpoint_interval: int = 1
    gradient_accumulation_steps: int = 1
    device: Optional[str] = None
    resume_dir: Optional[Path] = None
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
        """Deserialize a dictionary into a Config object."""
        return serde.from_dict(cls, cfg)
