##
##
##

import logging
from pathlib import Path


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
