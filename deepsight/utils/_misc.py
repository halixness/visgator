##
##
##

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def setup_logger(file: Path, debug: bool = False) -> logging.Logger:
    """Sets up the logger for the project.

    Parameters
    ----------
    file : Path
        The file to save the logs to.
    debug : bool, optional
        Whether to use debug mode or not. If debug mode is enabled,
        then the logger will log debug messages, otherwise it will
        log info messages.

    Returns
    -------
    logging.Logger
        The logger for the project.
    """

    logger = logging.getLogger("visgator")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False

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


def init_environment(seed: int, debug: bool) -> None:
    """Initializes the environment for reproducibility.

    Parameters
    ----------
    seed : int
        The seed to use for reproducibility. This seed is used for
        `random`, `numpy`, and `torch`.
    debug : bool
        Whether to use debug mode or not. If debug mode is enabled,
        then deterministic algorithms are used for `torch`.
    """

    torch.set_default_dtype(torch.float32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if debug:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        # just to be sure
        os.unsetenv("CUBLAS_WORKSPACE_CONFIG")
