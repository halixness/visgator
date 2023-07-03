##
##
##

from typing import Any

import torch.optim.lr_scheduler as lrs
from typing_extensions import Self

from visgator.lr_schedulers import LRScheduler as _LRSchduler
from visgator.optimizers import Optimizer

from ._config import Config


class LRScheduler(_LRSchduler):
    def __init__(
        self,
        config: Config,
        optimizer: Optimizer,
        num_epochs: int,
        steps_per_epoch: int,
    ) -> None:
        cls = getattr(lrs, config.name)
        self._step_after_batch = False
        if config.name == "OneCycleLR":
            config.args["total_steps"] = num_epochs * steps_per_epoch
            self._step_after_batch = True

        self._scheduler: lrs.LRScheduler = cls(optimizer, **config.args)
        self._steps_per_epoch = steps_per_epoch

    @classmethod
    def from_config(
        cls,
        config: Config,  # type: ignore
        optimizer: Optimizer,
        num_epochs: int,
        steps_per_epoch: int,
    ) -> Self:
        return cls(config, optimizer, num_epochs, steps_per_epoch)

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
        state = self._scheduler.state_dict()
        state.pop("anneal_func")

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.name == "OneCycleLR":
            last_epoch = state_dict["last_epoch"]
            epoch = last_epoch // self._steps_per_epoch
            state_dict["last_epoch"] = epoch * self._steps_per_epoch

        self._scheduler.load_state_dict(state_dict)
