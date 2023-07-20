##
##
##

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Sample:
    path: Path
    bbox: tuple[float, float, float, float]
    description: str
