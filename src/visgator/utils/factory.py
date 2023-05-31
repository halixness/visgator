##
##
##

import importlib
from typing import Type, TypeVar

_T = TypeVar("_T")


def get_subclass(parent: Type[_T], submodule: str) -> Type[_T]:
    parent_module = ".".join(parent.__module__.split(".")[:-1])
    module = importlib.import_module(f"{parent_module}.{submodule}")
    sub_cls = getattr(module, parent.__name__)
    if not issubclass(sub_cls, parent):
        raise ValueError(
            f"Expected subclass of {parent.__name__} but got {sub_cls.__name__}."
        )
    return sub_cls  # type: ignore
