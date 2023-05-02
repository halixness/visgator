##
##
##

import os

import torch


def init_torch(seed: int, debug: bool) -> None:
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True

    if debug:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        try:
            del os.environ["CUBLAS_WORKSPACE_CONFIG"]
            # just to be sure
            os.unsetenv("CUBLAS_WORKSPACE_CONFIG")
        except KeyError:
            pass
