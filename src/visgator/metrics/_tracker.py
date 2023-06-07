##
##
##

from dataclasses import dataclass
from typing import Generic, TypeVar

import torchmetrics as tm
from jaxtyping import Float
from torch import Tensor, nn

from visgator.models import LossInfo

_T = TypeVar("_T")


@dataclass(frozen=True)
class LossStatistics(Generic[_T]):
    total: _T
    unscaled: dict[str, _T]
    scaled: dict[str, _T]

    def items(self) -> list[tuple[str, _T]]:
        return [
            ("total", self.total),
            *[(f"unscaled/{name}", value) for name, value in self.unscaled.items()],
            *[(f"scaled/{name}", value) for name, value in self.scaled.items()],
        ]


class LossTracker(nn.Module):
    def __init__(self, losses: list[LossInfo]):
        super().__init__()

        self._weights = {loss.name: loss.weight for loss in losses}

        self._unscaled = nn.ModuleDict(
            {
                loss.name: tm.MetricTracker(tm.MeanMetric("error"), maximize=False)
                for loss in losses
            }
        )

        self._scaled = nn.ModuleDict(
            {
                loss.name: tm.MetricTracker(tm.MeanMetric("error"), maximize=False)
                for loss in losses
            }
        )

        self._total = tm.MetricTracker(tm.MeanMetric("error"), maximize=False)

    @property
    def n_steps(self) -> int:
        return self._total.n_steps

    def increment(self) -> None:
        tracker: tm.MetricTracker

        for tracker in self._unscaled.values():
            tracker.increment()

        for tracker in self._scaled.values():
            tracker.increment()

        self._total.increment()

    def update(self, losses: dict[str, Float[Tensor, ""]]) -> None:
        scaled_losses: list[Float[Tensor, ""]] = []

        tracker: tm.MetricTracker
        for name, loss in losses.items():
            tracker = self._unscaled[name]
            tracker.update(loss)

            tracker = self._scaled[name]
            value = loss * self._weights[name]
            scaled_losses.append(value)
            tracker.update(value)

        self._total.update(sum(scaled_losses))

    def compute(self) -> LossStatistics[Float[Tensor, ""]]:
        return LossStatistics(
            total=self._total.compute(),
            unscaled={
                name: tracker.compute() for name, tracker in self._unscaled.items()
            },
            scaled={name: tracker.compute() for name, tracker in self._scaled.items()},
        )

    def compute_all(self) -> LossStatistics[Float[Tensor, "N"]]:  # noqa: F821
        return LossStatistics(
            total=self._total.compute_all(),  # type: ignore
            unscaled={
                name: tracker.compute_all() for name, tracker in self._unscaled.items()
            },
            scaled={
                name: tracker.compute_all() for name, tracker in self._scaled.items()
            },
        )

    def best_loss(self) -> LossStatistics[tuple[float, int]]:
        return LossStatistics(
            total=self._total.best_metric(True),  # type: ignore
            unscaled={
                name: tracker.best_metric(True)
                for name, tracker in self._unscaled.items()
            },
            scaled={
                name: tracker.best_metric(True)
                for name, tracker in self._scaled.items()
            },
        )

    def forward(
        self, losses: dict[str, Float[Tensor, ""]]
    ) -> LossStatistics[Float[Tensor, ""]]:
        unscaled = {}
        scaled = {}

        tracker: tm.MetricTracker
        for name, loss in losses.items():
            tracker = self._unscaled[name]
            unscaled[name] = tracker(loss)

            tracker = self._scaled[name]
            value = tracker(loss * self._weights[name])
            scaled[name] = value

        total = self._total(sum(scaled.values()))

        return LossStatistics(total=total, unscaled=unscaled, scaled=scaled)

    def __call__(
        self, losses: dict[str, Float[Tensor, ""]]
    ) -> LossStatistics[Float[Tensor, ""]]:
        return super().__call__(losses)  # type: ignore

    def reset(self) -> None:
        self._total.reset()

        tracker: tm.MetricTracker
        for tracker in self._unscaled.values():
            tracker.reset()

        for tracker in self._scaled.values():
            tracker.reset()

    def reset_all(self) -> None:
        self._total.reset_all()

        tracker: tm.MetricTracker
        for tracker in self._unscaled.values():
            tracker.reset_all()

        for tracker in self._scaled.values():
            tracker.reset_all()

    def items(self) -> list[tuple[str, tm.MetricTracker]]:
        return [
            ("total", self._total),
            *[
                (f"unscaled/{name}", tracker)
                for name, tracker in self._unscaled.items()
            ],
            *[(f"scaled/{name}", tracker) for name, tracker in self._scaled.items()],
        ]
