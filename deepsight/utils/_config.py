##
##
##


import abc
import importlib
import json
from pathlib import Path
from typing import Any, Type, TypeVar

from ruamel.yaml import YAML
from typing_extensions import Self

T = TypeVar("T")


class Config(abc.ABC):
    """Abstract base class for all configs."""

    @property
    def module(self) -> str:
        return get_public_parent_module(cls=type(self))

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        """Deserializes a config object from a dictionary.

        Parameters
        ----------
        cfg : dict[str, Any]
            The serialized config.

        Returns
        -------
        Self
            The deserialized config.
        """

        module_path = cfg.get("module")
        file_path = cfg.get("file")

        if (module_path is not None and file_path is not None) or (
            module_path is None and file_path is None
        ):
            raise ValueError("Dataset config must have either a module or file field.")

        if file_path is not None:
            return cls.from_file(path=Path(file_path))

        module_path = str(cfg.pop("module"))
        if module_path.startswith("."):
            base_module = get_public_parent_module(cls).split(".")
            module_path = module_path[1:]

            while module_path.startswith("."):
                base_module.pop()
                module_path = module_path[1:]

            module_path = ".".join(base_module) + module_path

        module = importlib.import_module(module_path)
        sub_cls = getattr(module, cls.__name__)
        return sub_cls.from_dict(cfg=cfg)  # type: ignore

    @classmethod
    def from_file(cls, path: Path) -> Self:
        match path.suffix:
            case ".json":
                with path.open("r") as f:
                    cfg = json.load(f)
                return cls.from_dict(cfg=cfg)
            case ".yaml" | ".yml":
                yaml = YAML(typ="safe")
                cfg = yaml.load(path)
                return cls.from_dict(cfg=cfg)
            case ".py":
                path = path.with_suffix("")
                module_path = str(path).replace("/", ".")
                module = importlib.import_module(module_path)
                if not hasattr(module, "config"):
                    raise ValueError(f"Config module {module} has no config attribute.")

                return getattr(module, "config")  # type: ignore
            case _:
                raise ValueError(f"Unsupported file extension: {path.suffix}")

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serializes the config to a dictionary.

        Returns
        -------
        dict[str, Any]
            The serialized config.
        """


def get_public_parent_module(cls: Type[T]) -> str:
    module = cls.__module__.split(".")
    while module[-1].startswith("_"):
        module.pop()

    return ".".join(module)
