##
##
##

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import serde
from typing_extensions import Self

from deepsight.data.dataset import Config as DatasetConfig
from deepsight.lr_schedulers import Config as LRConfig
from deepsight.modeling.pipeline import Config as PipelineConfig
from deepsight.optimizers import Config as OptimizerConfig
from deepsight.utils import Config as _Config
from deepsight.utils import wandb
from deepsight.utils.torch import FloatType


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True, slots=True)
class Params:
    num_epochs: int
    train_batch_size: int
    eval_batch_size: int

    dataset: DatasetConfig = serde.field(
        serializer=lambda x: x.to_dict(),
        deserializer=DatasetConfig.from_dict,
    )
    pipeline: PipelineConfig = serde.field(
        serializer=lambda x: x.to_dict(),
        deserializer=PipelineConfig.from_dict,
    )
    optimizer: OptimizerConfig = serde.field(
        serializer=lambda x: x.to_dict(),
        deserializer=OptimizerConfig.from_dict,
    )
    lr_scheduler: LRConfig = serde.field(
        serializer=lambda x: x.to_dict(),
        deserializer=LRConfig.from_dict,
    )

    seed: int = 3407
    # compile: bool = True
    checkpoint_interval: int = 1
    eval_interval: int = 1
    gradient_accumulation_steps: int = 1
    device: str | None = None
    dtype: FloatType = FloatType.FLOAT32
    init_scale: int | float | None = None
    max_grad_norm: float | None = None

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
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)


@serde.serde()
@dataclass(slots=True)
class Config(_Config):
    """Configuration for the trainer engine.

    Attributes
    ----------
    dir : Path
        The direcotry where to save the results. If such direvtory already
        exists and a previous run is found, the run will be resumed.
    wandb : WandbConfig
        Configuration for the Weights & Biases integration.
    params : Params | None
        The parameters to use for the training. If the run is resumed, this field
        can be set to `None` since the parameters will be loaded from the previous
        run. Defaults to `None`.
    debug : bool
        Whether to run the engine in debug mode. Defaults to `False`.
    """

    dir: Path = serde.field(
        default=Path("output"),
        serializer=lambda x: str(x),
    )
    wandb: wandb.Config = serde.field(
        default=wandb.Config(job_type="train"),
        serializer=wandb.Config.to_dict,
        deserializer=wandb.Config.from_dict,
    )
    params: Params | None = serde.field(
        default=None,
        serializer=lambda x: x.to_dict() if x is not None else None,
        deserializer=Params.from_dict,
    )
    debug: bool = False

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)
