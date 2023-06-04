##
##
##

import importlib
import logging
from pathlib import Path
from typing import Type, TypeVar


def setup_logger(file: Path, debug: bool = False) -> logging.Logger:
    logger = logging.getLogger("visgator")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False

    # for handler in logger.handlers:
    #     logger.removeHandler(handler)
    # for filter in logger.filters:
    #     logger.removeFilter(filter)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


_T = TypeVar("_T")


def fullname(cls: Type[_T]) -> str:
    return cls.__module__ + "." + cls.__name__


def get_subclass(module_path: str, cls: Type[_T]) -> Type[_T]:
    if module_path.startswith("."):
        if len(module_path) == 1:
            module_path = public_parent_module(cls)
        else:
            module_path = public_parent_module(cls) + module_path

    module = importlib.import_module(module_path)
    sub_cls = getattr(module, cls.__name__)
    if not issubclass(sub_cls, cls):
        raise ValueError(
            f"Expected subclass of {fullname(cls)} but got {fullname(sub_cls)}."
        )

    return sub_cls  # type: ignore


def public_parent_module(cls: Type[_T]) -> str:
    module = cls.__module__.split(".")
    while module[-1].startswith("_"):
        module.pop()

    return ".".join(module)
