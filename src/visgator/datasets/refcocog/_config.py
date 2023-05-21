##
##
##

"""Configurations for RefCOCO dataset."""

import enum
from pathlib import Path
from typing import Any

from typing_extensions import Self

from .._config import Config as BaseConfig


class SplitProvider(enum.Enum):
    GOOGLE = "google"
    UMD = "umd"

    @classmethod
    def from_str(cls, split: str) -> Self:
        return cls[split.upper().strip()]

    def __str__(self) -> str:
        return self.value


class Config(BaseConfig):
    """Configuration for RefCOCOg dataset."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)

        path = cfg.get("path", None)
        if path is None:
            raise ValueError("Missing 'path' field in RefCOCO config.")
        self._path = Path(path)

        self._split_provider = SplitProvider.from_str(cfg.get("split_provider", "umd"))

    @property
    def path(self) -> Path:
        """Returns the path to the dataset."""
        return self._path

    @property
    def split_provider(self) -> SplitProvider:
        """Returns the split provider."""
        return self._split_provider
