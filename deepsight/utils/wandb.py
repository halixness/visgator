##
##
##

from dataclasses import dataclass
from typing import Any

import serde
from typing_extensions import Self


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True, slots=True)
class Config:
    """Configuration for the Weights & Biases integration.

    Attributes
    ----------
    enabled : bool
        Whether to enable the integration. Defaults to `True`.
    project : str | None
        The name of the project to use. Defaults to `None`.
    job_type : str | None
        The type of job option. Defaults to `None`.
    entity : str | None
        The name of the entity. Defaults to `None`.
    name : str | None
        The name of the run. Defaults to `None`.
    tags : list[str] | None
        The tags to use. Defaults to `None`.
    notes : str | None
        The notes to use. Defaults to `None`.
    id : str | None
        The ID of the run to resume. Defaults to `None`.
    save : bool
        Whether to copy all the generated files to the cloud. Defaults to `False`.
    """

    enabled: bool = True
    project: str | None = serde.field(default=None, skip_if=lambda x: x is None)
    job_type: str | None = serde.field(default=None, skip_if=lambda x: x is None)
    entity: str | None = serde.field(default=None, skip_if=lambda x: x is None)
    name: str | None = serde.field(default=None, skip_if=lambda x: x is None)
    tags: list[str] | None = serde.field(default=None, skip_if=lambda x: x is None)
    notes: str | None = serde.field(default=None, skip_if=lambda x: x is None)
    id: str | None = serde.field(default=None, skip_if=lambda x: x is None)
    save: bool = False

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        return serde.from_dict(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)
