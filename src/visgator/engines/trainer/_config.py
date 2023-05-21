##
##
##

from pathlib import Path
from typing import Any, Optional

from typing_extensions import Self

from visgator.datasets import Config as DatasetConfig
from visgator.models import Config as ModelConfig

from .lr_schedulers import Config as LRSchedulerConfig
from .optimizers import Config as OptimizerConfig


def check_field(cfg: dict[str, Any], field: str) -> Any:
    if field not in cfg:
        raise ValueError(f"Missing field '{field}' in training configuration.")
    return cfg[field]


class Config:
    def __init__(self, cfg: dict[str, Any]) -> None:
        # configurations with default values
        self._seed = int(cfg.get("seed", 3407))
        self._debug = bool(cfg.get("debug", False))
        self._compile = bool(cfg.get("compile", False))
        self._mixed_precision = bool(cfg.get("mixed_precision", True))
        self._output_dir = Path(cfg.get("output_dir", "output"))
        self._checkpoint_interval = int(cfg.get("checkpoint_interval", 1))
        self._gradient_accumulation_steps = int(
            cfg.get("gradient_accumulation_steps", 1)
        )

        if self._gradient_accumulation_steps < 1:
            raise ValueError(
                "Invalid gradient accumulation steps: "
                f"{self._gradient_accumulation_steps}. Must be greater than 0."
            )

        # optional configurations
        self._device = None
        if "device" in cfg:
            self._device = str(cfg["device"])

        self._resume_dir = None
        if "resume_dir" in cfg:
            self._resume_dir = Path(cfg.get("resume_dir"))  # type: ignore
            if "output_dir" in cfg:
                print("Warning: Ignoring 'output_dir' when resuming training.")

        self._max_grad_norm = None
        if "max_grad_norm" in cfg:
            self._max_grad_norm = float(cfg["max_grad_norm"])
            if self._max_grad_norm <= 0.0:
                raise ValueError(
                    f"Invalid max gradient norm: {self._max_grad_norm}. "
                    "Must be greater than 0."
                )

        # required configurations
        self._num_epochs = int(check_field(cfg, "num_epochs"))
        self._batch_size = int(check_field(cfg, "batch_size"))
        if self._batch_size < 1:
            raise ValueError(
                f"Invalid batch size: {self._batch_size}. Must be greater than 0."
            )
        if self._batch_size % self._gradient_accumulation_steps != 0:
            raise ValueError(
                f"Invalid batch size: {self._batch_size}. Must be divisible by "
                f"gradient accumulation steps: {self._gradient_accumulation_steps}."
            )

        self._dataset = DatasetConfig.from_dict(check_field(cfg, "dataset"))
        self._model = ModelConfig.from_dict(check_field(cfg, "model"))
        self._optimizer = OptimizerConfig.from_dict(check_field(cfg, "optimizer"))
        self._lr_scheduler = LRSchedulerConfig.from_dict(
            check_field(cfg, "lr_scheduler")
        )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return cls(cfg)

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def debug(self) -> bool:
        return self._debug

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def resume_dir(self) -> Optional[Path]:
        return self._resume_dir

    @property
    def checkpoint_interval(self) -> int:
        return self._checkpoint_interval

    @property
    def device(self) -> Optional[str]:
        return self._device

    @property
    def num_epochs(self) -> int:
        return self._num_epochs

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def mixed_precision(self) -> bool:
        return self._mixed_precision

    @property
    def max_grad_norm(self) -> Optional[float]:
        return self._max_grad_norm

    @property
    def gradient_accumulation_steps(self) -> int:
        return self._gradient_accumulation_steps

    @property
    def dataset(self) -> DatasetConfig:
        return self._dataset

    @property
    def model(self) -> ModelConfig:
        return self._model

    @property
    def compile(self) -> bool:
        return self._compile

    @property
    def optimizer(self) -> OptimizerConfig:
        return self._optimizer

    @property
    def lr_scheduler(self) -> LRSchedulerConfig:
        return self._lr_scheduler
