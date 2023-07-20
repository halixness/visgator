##
##
##

from dataclasses import dataclass
from typing import Any

from typing_extensions import Self

from deepsight.lr_schedulers import Config as _Config


@dataclass(frozen=True)
class Config(_Config):
    """Configuration for PyTorch learning rate schedulers."""

    name: str
    args: dict[str, Any]

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        name = str(cfg["name"])
        args = dict(cfg["args"])
        return cls(name, args)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
        }
