##
##
##

import importlib
from typing import Any, Type, TypeVar

_T = TypeVar("_T")


def instantiate(
    class_path: str,
    parent_class: Type[_T],
    *args: Any,
    **kwargs: Any,
) -> _T:
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    if not issubclass(cls, parent_class):
        raise TypeError(f"{cls} is not a subclass of {parent_class}.")

    return cls(*args, **kwargs)  # type: ignore
