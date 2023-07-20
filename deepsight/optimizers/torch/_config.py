##
##
##

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import serde
from typing_extensions import Self

from deepsight.optimizers import Config as _Config


@dataclass(frozen=True)
class ParamGroupConfig:
    """Configuration for a parameter group.

    Parameters
    ----------
    regex : str
        The regular expression to match parameter names.
    args : dict[str, Union[str, bool, int, float]]
        The arguments to pass to the PyTorch optimizer.
    """

    regex: str
    args: dict[str, Any]

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        regex = cfg["regex"]
        args = cfg["args"]
        return cls(regex, args)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class Config(_Config):
    """Configuration for PyTorch optimizers.

    Parameters
    ----------
    name : str
        The name of the PyTorch optimizer.
    groups : list[ParamGroup], optional
        The parameter groups to pass to the PyTorch optimizer.
    freeze : list[str], optional
        The regexes of the parameters to freeze.
    """

    name: str
    groups: list[ParamGroupConfig]
    freeze: list[str] = serde.field(default_factory=list)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)
