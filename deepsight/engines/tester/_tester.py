##
##
##

import json
import logging
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
from typing import Generic, TypeVar

import torch
import torchmetrics as tm
import wandb
from torch import autocast  # type: ignore
from tqdm import tqdm
from typing_extensions import Self

from deepsight.data.dataset import DataLoader, Dataset, Split
from deepsight.data.structs import BoundingBoxes, RECInput, RECOutput
from deepsight.measures.metrics import BoxIoU, BoxIoUAccuracy, GeneralizedBoxIoU
from deepsight.modeling.pipeline import Pipeline
from deepsight.optimizers import Optimizer
from deepsight.utils import init_environment, setup_logger
from deepsight.utils.torch import Device

from ._config import Config

ModelInput = TypeVar("ModelInput")
ModelOutput = TypeVar("ModelOutput")


class Tester(Generic[ModelInput, ModelOutput]):
    def __init__(self, config: Config) -> None:
        self._config = config

        # set in the following order
        self._dir: Path
        self._logger: logging.Logger

        self._device: Device
        self._loader: DataLoader[RECInput, RECOutput]
        self._pipeline: Pipeline[RECInput, RECOutput, ModelInput, ModelOutput]
        self._optimizer: Optimizer

    @classmethod
    def new(cls, config: Config) -> Self:
        return cls(config)

    def _setup_launch(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._dir = self._config.dir / now
        self._dir.mkdir(parents=True, exist_ok=False)

        wandb_cfg = self._config.wandb
        if wandb_cfg.enabled:
            wandb.init(
                project=wandb_cfg.project,
                entity=wandb_cfg.entity,
                job_type=wandb_cfg.job_type,
                name=wandb_cfg.name,
                tags=wandb_cfg.tags,
                notes=wandb_cfg.notes,
                dir=self._dir,
            )
        else:
            wandb.init(mode="disabled")

    def _set_device(self) -> None:
        self._device = Device(self._config.device)
        self._logger.info(f"Using device {self._device}.")

    def _save_config(self) -> None:
        config_file = self._dir / "config.json"
        config = self._config.to_dict()
        wandb.config.update(config)

        with config_file.open("w") as f:
            json.dump(config, f, indent=2)

    def _set_loaders(self) -> None:
        """Sets the test data loader."""

        dataset = Dataset.new_for_rec(
            self._config.dataset,
            split=Split.TEST,
            debug=self._config.debug,
        )

        self._loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=1,
        )

        self._logger.info(f"Using {dataset.name} dataset.")
        self._logger.info(f"\tsize: {len(dataset)}")
        self._logger.info(f"\tbatch size: {1}")

    def _set_pipeline(self) -> None:
        pipeline: Pipeline[RECInput, RECOutput, ModelInput, ModelOutput]

        pipeline = Pipeline.new_for_rec(self._config.pipeline)
        pipeline = pipeline.to(self._device.to_torch())

        self._logger.info(f"Using {pipeline.name} pipeline.")

        if self._config.weights is None:
            self._logger.warning("No weights provided, testing with initial weights.")
        else:
            try:
                checkpoint = torch.load(
                    self._config.weights, map_location=self._device.to_torch()
                )

                if "pipeline" in checkpoint:
                    checkpoint = checkpoint["pipeline"]

                pipeline.load_state_dict(checkpoint)
                self._logger.info("Loaded pipeline weights.")
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find weights at {self._config.weights}."
                )

        self._pipeline = pipeline

    @torch.no_grad()
    def _run(self) -> None:
        self._logger.info("Testing started.")

        metrics = tm.MetricCollection(
            {
                "IoU": BoxIoU(),
                "GIoU": GeneralizedBoxIoU(),
                "Accuracy@50": BoxIoUAccuracy(0.5),
                "Accuracy@75": BoxIoUAccuracy(0.75),
                "Accuracy@90": BoxIoUAccuracy(0.9),
            },
            compute_groups=False,
        ).to(self._device.to_torch())

        start = timer()

        self._pipeline.eval()

        device_type = "cuda" if self._device.is_cuda else "cpu"
        for inputs, outputs in tqdm(self._loader, desc="Testing"):
            inputs = inputs.to(self._device.to_torch())
            outputs = outputs.to(self._device.to_torch())

            with autocast(
                device_type,
                enabled=self._config.dtype.is_mixed_precision(),
                dtype=self._config.dtype.to_torch_dtype(),
            ):
                model_input = self._pipeline.preprocessor(inputs, None)
                model_output = self._pipeline.model(model_input)

            predictions = self._pipeline.postprocessor(model_output)
            pred_boxes = BoundingBoxes.stack(
                [prediction.box for prediction in predictions]
            )
            tgt_boxes = BoundingBoxes.stack([output.box for output in outputs])
            metrics.update(pred_boxes, tgt_boxes)

        end = timer()
        elapsed = end - start

        self._logger.info("Testing finished.")
        self._logger.info("Statistics:")
        self._logger.info(f"\telapsed time: {elapsed:.2f} s.")
        self._logger.info(
            f"\ttime per sample: {elapsed / len(self._loader.dataset):.2f} s."
        )

        self._logger.info("\tmetrics:")
        columns = []
        row = []
        for name, metric in metrics.items():
            columns.append(name)
            value = metric.compute()
            row.append(value)
            self._logger.info(f"\t\t{name}: {value:.5f}")

        table = wandb.Table(columns=columns, data=[row])
        wandb.log({"metrics": table})

    def run(self) -> None:
        self._setup_launch()
        try:
            self._logger = setup_logger(self._dir / "test.log", self._config.debug)
            init_environment(self._config.seed, self._config.debug)
            # TODO: solve issues with pyserde serialization
            # self._save_config()
            self._set_device()

            self._set_loaders()
            self._set_pipeline()

            self._run()

        except Exception as e:
            self._logger.error(f"Testing failed with the following error: {e}")
            raise

        wandb.finish()
