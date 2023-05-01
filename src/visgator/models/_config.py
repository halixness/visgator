##
##
##

from __future__ import annotations

from typing import Any

from visgator.utils import instantiate


class Config:
    def __init__(self, cfg: dict[str, Any]) -> None:
        name = cfg.get("name", None)
        if name is None:
            raise ValueError("Missing 'name' field in model configuration.")
        self._name = str(name)

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def from_dict(cfg: dict[str, Any]) -> Config:
        child_module = str(cfg["name"]).lower()
        parent_module = ".".join(Config.__module__.split(".")[:-1])
        module = f"{parent_module}.{child_module}"
        class_path = f"{module}.Config"

        return instantiate(class_path, Config, cfg)  # type: ignore
