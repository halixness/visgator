##
##
##

from ._config import Config


class Trainer:
    def __init__(self, config: Config) -> None:
        self._config = config

    def run(self) -> None:
        raise NotImplementedError
