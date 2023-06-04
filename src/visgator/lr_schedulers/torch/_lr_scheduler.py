##
##
##

from typing import Any

import torch
from typing_extensions import Self

from visgator.lr_schedulers import LRScheduler as _LRSchduler
from visgator.optimizers import Optimizer

from ._config import Config


class LRScheduler(_LRSchduler):
    def __init__(self, config: Config, optimizer: Optimizer) -> None:
        cls = getattr(torch.optim.lr_scheduler, config.name)
        self._scheduler: torch.optim.lr_scheduler.LRScheduler = cls(
            optimizer, **config.args
        )

        if config.name == "OneCycleLR":
            self._step_after_batch = True
        else:
            self._step_after_batch = False

    @classmethod
    def from_config(cls, config: Config, optimizer: Optimizer) -> Self:  # type: ignore
        return cls(config, optimizer)

    @property
    def name(self) -> str:
        return self._scheduler.__class__.__name__

    def step_after_epoch(self) -> None:
        if not self._step_after_batch:
            self._scheduler.step()

    def step_after_batch(self) -> None:
        if self._step_after_batch:
            self._scheduler.step()

    @property
    def last_lr(self) -> float:
        return self._scheduler.get_last_lr()[-1]

    def state_dict(self) -> dict[str, Any]:
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._scheduler.load_state_dict(state_dict)
