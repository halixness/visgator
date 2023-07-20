##
##
##

import torchmetrics as tm
from jaxtyping import Float
from torch import Tensor, nn

from deepsight.measures import Loss, Losses


class LossTracker(nn.Module):
    """Module that tracks losses."""

    def __init__(self, losses_names: list[str]) -> None:
        super().__init__()

        self._unscaled = nn.ModuleDict(
            {
                name: tm.MetricTracker(tm.MeanMetric("error"), maximize=False)
                for name in losses_names
            }
        )

        self._scaled = nn.ModuleDict(
            {
                name: tm.MetricTracker(tm.MeanMetric("error"), maximize=False)
                for name in losses_names
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

    def update(self, losses: list[Loss]) -> None:
        scaled_losses: list[Tensor] = []

        tracker: tm.MetricTracker
        for loss in losses:
            tracker = self._unscaled[loss.name]
            tracker.update(loss.value)

            tracker = self._scaled[loss.name]
            value = loss.value * loss.weight
            tracker.update(value)
            scaled_losses.append(value)

        self._total.update(sum(scaled_losses))

    def compute(self) -> Losses[Float[Tensor, ""]]:
        return Losses(
            total=self._total.compute(),
            unscaled={
                name: tracker.compute() for name, tracker in self._unscaled.items()
            },
            scaled={name: tracker.compute() for name, tracker in self._scaled.items()},
        )

    def compute_all(self) -> Losses[Float[Tensor, "N"]]:  # noqa: F821
        return Losses(
            total=self._total.compute(),
            unscaled={
                name: tracker.compute_all() for name, tracker in self._unscaled.items()
            },
            scaled={
                name: tracker.compute_all() for name, tracker in self._scaled.items()
            },
        )

    def forward(self, losses: list[Loss]) -> Losses[Float[Tensor, ""]]:
        unscaled = {}
        scaled = {}

        tracker: tm.MetricTracker
        for loss in losses:
            tracker = self._unscaled[loss.name]
            unscaled[loss.name] = tracker(loss.value)

            tracker = self._scaled[loss.name]
            value = tracker(loss.value * loss.weight)
            scaled[loss.name] = value

        total = self._total(sum(scaled.values()))

        return Losses(total=total, unscaled=unscaled, scaled=scaled)

    def __call__(self, losses: list[Loss]) -> Losses[Float[Tensor, ""]]:
        return super().__call__(losses)  # type: ignore

    def best_loss(self) -> Losses[tuple[float, int]]:
        return Losses(
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
