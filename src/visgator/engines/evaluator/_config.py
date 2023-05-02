##
##
##

from pathlib import Path
from typing import Any, Optional, Self

from visgator.datasets import Config as DatasetConfig
from visgator.models import Config as ModelConfig


def check_field(cfg: dict[str, Any], field: str) -> Any:
    if field not in cfg:
        raise ValueError(f"Missing field '{field}' in evaluation configuration.")
    return cfg[field]


class Config:
    def __init__(self, cfg: dict[str, Any]) -> None:
        # configurations with default values
        self._seed = int(cfg.get("seed", 3407))
        self._debug = bool(cfg.get("debug", False))
        self._compile = bool(cfg.get("compile", False))
        self._output_dir = Path(cfg.get("output_dir", "./output"))

        # optional configurations
        self._device = None
        if "device" in cfg:
            self._device = str(cfg["device"])

        self._weights: Optional[Path] = None
        weights = cfg.get("weights", None)
        if weights is not None:
            self._weights = Path(weights)

        # required configurations
        self._dataset = DatasetConfig.from_dict(check_field(cfg, "dataset"))
        self._model = ModelConfig.from_dict(check_field(cfg, "model"))

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return cls(cfg)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def debug(self) -> bool:
        return self._debug

    @property
    def device(self) -> Optional[str]:
        return self._device

    @property
    def compile(self) -> bool:
        return self._compile

    @property
    def dataset(self) -> DatasetConfig:
        return self._dataset

    @property
    def weights(self) -> Optional[Path]:
        return self._weights

    @property
    def model(self) -> ModelConfig:
        return self._model
