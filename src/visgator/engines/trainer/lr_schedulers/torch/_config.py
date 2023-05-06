##
##
##

from typing import Any

from .._config import Config as BaseConfig
from .._config import Provider


class Config(BaseConfig):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)

        self._args = dict(cfg.get("args", {}))

    @property
    def args(self) -> dict[str, Any]:
        return self._args

    @property
    def provider(self) -> Provider:
        return Provider.TORCH
