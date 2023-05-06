##
##
##

from typing import Optional

import torchmetrics as tm
from jaxtyping import Float
from torch import Tensor, nn

from visgator.models import LossInfo


class LossTracker(nn.Module):
    def __init__(self, losses: list[LossInfo]):
        super().__init__()
        self._weights = {loss.name: loss.weight for loss in losses}

        self._total_loss = tm.MetricTracker(tm.MeanMetric("error"), maximize=False)

        self._unscaled = nn.ModuleDict()
        self._scaled = nn.ModuleDict()

        for loss in losses:
            self._unscaled.add_module(
                f"{loss.name}_unscaled",
                tm.MetricTracker(tm.MeanMetric("error"), maximize=False),
            )

            self._scaled.add_module(
                f"{loss.name}_scaled",
                tm.MetricTracker(tm.MeanMetric("error"), maximize=False),
            )

    def add_loss(self, loss: LossInfo) -> None:
        self._weights[loss.name] = loss.weight

        self._unscaled.add_module(
            f"{loss.name}_unscaled",
            tm.MetricTracker(tm.MeanMetric("error"), maximize=False),
        )

        self._scaled.add_module(
            f"{loss.name}_scaled",
            tm.MetricTracker(tm.MeanMetric("error"), maximize=False),
        )

    @property
    def n_steps(self) -> int:
        return self._total_loss.n_steps

    def increment(self) -> None:
        self._total_loss.increment()

        tracker: tm.MetricTracker
        for tracker in self._unscaled.values():
            tracker.increment()

        for tracker in self._scaled.values():
            tracker.increment()

    def update(self, losses: dict[str, Float[Tensor, ""]]) -> None:
        scaled_losses: list[Float[Tensor, ""]] = []

        tracker: tm.MetricTracker
        for name, loss in losses.items():
            tracker = self._unscaled[f"{name}_unscaled"]
            tracker.update(loss)

            tracker = self._scaled[f"{name}_scaled"]
            value = loss * self._weights[name]
            scaled_losses.append(value)
            tracker.update(value)

        self._total_loss.update(sum(scaled_losses))

    def compute(self) -> dict[str, Float[Tensor, ""]]:
        return {
            "total_loss": self._total_loss.compute(),
            **{name: tracker.compute() for name, tracker in self._unscaled.items()},
            **{name: tracker.compute() for name, tracker in self._scaled.items()},
        }

    def compute_all(self) -> dict[str, Float[Tensor, "N"]]:  # noqa: F821
        return {
            "total_loss": self._total_loss.compute_all(),  # type: ignore
            **{name: tracker.compute_all() for name, tracker in self._unscaled.items()},
            **{name: tracker.compute_all() for name, tracker in self._scaled.items()},
        }

    def best_loss(
        self, return_step: bool = False
    ) -> tuple[dict[str, float], Optional[dict[str, int]]]:
        values = {
            "total_loss": self._total_loss.best_metric(False),
            **{
                name: tracker.best_metric(False)
                for name, tracker in self._unscaled.items()
            },
            **{
                name: tracker.best_metric(False)
                for name, tracker in self._scaled.items()
            },
        }

        if not return_step:
            return values, None  # type: ignore

        epochs = {
            "total_loss": self._total_loss.best_metric(True)[1],  # type: ignore
            **{
                name: tracker.best_metric(True)[1]
                for name, tracker in self._unscaled.items()
            },
            **{
                name: tracker.best_metric(True)[1]
                for name, tracker in self._scaled.items()
            },
        }

        return values, epochs  # type: ignore

    def forward(
        self, losses: dict[str, Float[Tensor, ""]]
    ) -> dict[str, Float[Tensor, ""]]:
        scaled_losses: list[Float[Tensor, ""]] = []

        res = {}

        tracker: tm.MetricTracker
        for name, loss in losses.items():
            tracker = self._unscaled[f"{name}_unscaled"]
            res[f"{name}_unscaled"] = tracker(loss)

            tracker = self._scaled[f"{name}_scaled"]
            value = tracker(loss * self._weights[name])
            res[f"{name}_scaled"] = value
            scaled_losses.append(value)

        res["total_loss"] = self._total_loss(sum(scaled_losses))

        return res

    def __call__(
        self, losses: dict[str, Float[Tensor, ""]]
    ) -> dict[str, Float[Tensor, ""]]:
        return super().__call__(losses)  # type: ignore

    def reset(self) -> None:
        self._total_loss.reset()

        tracker: tm.MetricTracker
        for tracker in self._unscaled.values():
            tracker.reset()

        for tracker in self._scaled.values():
            tracker.reset()

    def reset_all(self) -> None:
        self._total_loss.reset_all()

        tracker: tm.MetricTracker
        for tracker in self._unscaled.values():
            tracker.reset_all()

        for tracker in self._scaled.values():
            tracker.reset_all()
