##
##
##

import logging
from datetime import datetime
from typing import Any

import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
from torchmetrics import MetricCollection
from tqdm import tqdm

from visgator.datasets import Dataset, Split
from visgator.metrics import GIoU, IoU
from visgator.models import Model
from visgator.utils.batch import Batch
from visgator.utils.bbox import BBoxes
from visgator.utils.device import Device
from visgator.utils.logging import setup_logger
from visgator.utils.misc import init_torch

from ._config import Config


class Evaluator:
    def __init__(self, config: Config) -> None:
        self._config = config

        # set in the following order
        self._logger: logging.Logger
        self._device: Device
        self._loader: DataLoader[tuple[Batch, BBoxes]]
        self._model: Model[Any]
        self._metrics: MetricCollection

    def _setup_environment(self) -> None:
        init_torch(self._config.seed, self._config.debug)

        if not self._config.output_dir.exists():
            self._config.output_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = self._config.output_dir / f"eval_{now}.log"
        self._logger = setup_logger(log_file, self._config.debug)

    def _set_device(self) -> None:
        if self._config.device is None:
            self._device = Device.default()
        else:
            self._device = Device.from_str(self._config.device)

        self._logger.info(f"Using device {self._device}.")

    def _set_dataloader(self) -> None:
        dataset = Dataset.from_config(
            self._config.dataset,
            split=Split.TEST,
            debug=self._config.debug,
        )
        sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=1, drop_last=False)

        self._loader = DataLoader(
            dataset,  # type: ignore
            batch_sampler=batch_sampler,
            num_workers=1,
            pin_memory=True,
            collate_fn=Dataset.batchify,
        )

        self._logger.info(f"Using dataset {self._config.dataset.name}:")
        self._logger.info(f"\tsize: {len(dataset)}")
        self._logger.info(f"\tbatch size: {1}")

    def _set_model(self) -> None:
        self._logger.info(f"Using model {self._config.model.name}.")

        model: Model[Any] = Model.from_config(self._config.model)
        model = model.to(self._device.to_torch())

        if self._config.weights is None:
            self._logger.warning("No weights loaded, using initialization weights.")
        else:
            try:
                weights = torch.load(
                    self._config.weights,
                    map_location=self._device.to_torch(),
                )
                model.load_state_dict(weights)
                self._logger.info(f"Loaded model weights from {self._config.weights}")
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"No weights found at '{self._config.weights}'."
                )

        if self._config.compile:
            self._logger.info("Compiling model.")
            self._model = torch.compile(model)  # type: ignore
        else:
            self._logger.info("Skipping model compilation.")
            self._model = model

    def _set_metrics(self) -> None:
        self._metrics = MetricCollection(
            {
                "IoU": IoU(),
                "GIoU": GIoU(),
            }
        ).to(self._device.to_torch())

    def _log_metrics(self) -> None:
        metrics = self._metrics.compute()

        self._logger.info("Metrics:")
        for name, value in metrics.items():
            self._logger.info(f"\t{name}: {value}")

    @torch.no_grad()
    def _eval(self) -> None:
        self._logger.info("Evaluation started.")

        self._model.eval()

        batch: Batch
        bboxes: BBoxes
        for batch, bboxes in tqdm(self._loader, desc="Evaluating"):
            batch = batch.to(self._device.to_torch())
            bboxes = bboxes.to(self._device.to_torch())

            output = self._model(batch)
            predictions = self._model.predict(output)

            self._metrics.update(predictions.xyxyn, bboxes.xyxyn)

        self._log_metrics()
        self._logger.info("Evaluation finished.")

    def run(self) -> None:
        try:
            self._setup_environment()
            self._set_device()
            self._set_dataloader()
            self._set_model()
            self._set_metrics()

            self._eval()

        except Exception as e:
            self._logger.error(f"Evaluation failed with the following error: {e}")
            raise
