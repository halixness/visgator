##
##
##

import enum
from pathlib import Path
from typing import Any, Optional, Self

from visgator.datasets import Config as DatasetConfig
from visgator.models import Config as ModelConfig


class Device(enum.Enum):
    CPU = "cpu"
    GPU = "gpu"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, s: str) -> Self:
        return cls[s.upper().strip()]


class Config:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._seed = int(cfg.get("seed", 3407))
        self._debug = bool(cfg.get("debug", False))
        self._device = Device.from_string(cfg.get("device", "gpu"))

        dataset = cfg.get("dataset", None)
        if dataset is None:
            raise ValueError("Missing 'dataset' field in config.")
        self._dataset = DatasetConfig.from_dict(dataset)

        self._weights: Optional[Path] = None
        weights = cfg.get("weights", None)
        if weights is not None:
            self._weights = Path(weights)

        model = cfg.get("model", None)
        if model is None:
            raise ValueError("Missing 'model' field in config.")
        self._model = ModelConfig.from_dict(model)

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def debug(self) -> bool:
        return self._debug

    @property
    def device(self) -> Device:
        return self._device

    @property
    def dataset(self) -> DatasetConfig:
        return self._dataset

    @property
    def weights(self) -> Optional[Path]:
        return self._weights

    @property
    def model(self) -> ModelConfig:
        return self._model
