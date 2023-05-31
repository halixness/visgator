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


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class Config:
    """Configuration for evaluation."""

    dataset: DatasetConfig = serde.field(deserializer=DatasetConfig.from_dict)
    model: ModelConfig = serde.field(deserializer=ModelConfig.from_dict)

    seed: int = 3407
    debug: bool = False
    compile: bool = False
    output_dir: Path = Path("output")
    device: Optional[str] = None
    weights: Optional[Path] = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Self:
        """Deserialize a dictionary into a Config object."""
        return serde.from_dict(cls, config)
