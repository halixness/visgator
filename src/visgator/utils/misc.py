##
##
##

import importlib
import os
from typing import Any, Type, TypeVar

import torch

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


def init_torch(seed: int, debug: bool) -> None:
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True

    if debug:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)  # type: ignore
        try:
            del os.environ["CUBLAS_WORKSPACE_CONFIG"]
            # just to be sure
            os.unsetenv("CUBLAS_WORKSPACE_CONFIG")
        except KeyError:
            pass
