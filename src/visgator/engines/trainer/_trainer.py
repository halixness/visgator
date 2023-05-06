##
##
##

import logging
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
from typing import Generic, Optional, TypeVar

import torch
import torchmetrics as tm
from torch import autocast  # type: ignore
from torch.cuda.amp import GradScaler  # type: ignore
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

from visgator.datasets import Dataset, Split
from visgator.metrics import GIoU, IoU, LossTracker
from visgator.models import Criterion, Model
from visgator.utils.batch import Batch
from visgator.utils.bbox import BBoxes
from visgator.utils.device import Device
from visgator.utils.logging import setup_logger
from visgator.utils.misc import init_torch

from ._checkpoint import Checkpoint
from ._config import Config
from .lr_schedulers import LRScheduler
from .optimizers import Optimizer

_T = TypeVar("_T")


class Trainer(Generic[_T]):
    def __init__(self, config: Config) -> None:
        self._config = config

        # set in the following order
        self._output_dir: Path
        self._logger: logging.Logger
        self._device: Device
        self._train_loader: DataLoader[tuple[Batch, BBoxes]]
        self._eval_loader: DataLoader[tuple[Batch, BBoxes]]
        self._model: Model[_T]
        self._criterion: Criterion[_T]
        self._optimizer: Optimizer
        self._lr_scheduler: LRScheduler
        self._scaler: GradScaler

        self._tl_tracker: LossTracker
        self._el_tracker: LossTracker
        self._tm_tracker: tm.MetricTracker
        self._em_tracker: tm.MetricTracker

    def _setup_environment(self) -> None:
        init_torch(self._config.seed, self._config.debug)

        if self._config.resume_dir is None:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._output_dir = self._config.output_dir / now
            self._output_dir.mkdir(parents=True, exist_ok=True)

            log_file = self._output_dir / f"train_{0}.log"
        else:
            self._output_dir = self._config.resume_dir

            log_files = list(self._output_dir.glob("*.log"))
            count = 0
            for log_file in log_files:
                count = max(count, int(log_file.stem[6:]))

            log_file = self._output_dir / f"train_{count + 1}.log"

        self._logger = setup_logger(log_file, self._config.debug)

    def _set_device(self) -> None:
        if self._config.device is None:
            self._device = Device.default()
        else:
            self._device = Device.from_str(self._config.device)

        self._logger.info(f"Using device {self._device}.")

    def _load_checkpoint(self) -> Optional[Checkpoint]:
        checkpoint_dir = self._output_dir / "checkpoints"
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            return None

        checkpoint_files = list(checkpoint_dir.glob("*.tar"))
        epoch = 0
        for checkpoint_file in checkpoint_files:
            epoch = max(epoch, int(checkpoint_file.stem[10:]))

        if epoch == 0:
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_dir}'.")

        try:
            checkpoint_file = checkpoint_dir / f"checkpoint_{epoch}.tar"
            checkpoint = Checkpoint.from_file(checkpoint_file, self._device.to_torch())
            self._logger.info(f"Loaded checkpoint from '{checkpoint_file}'.")
            return checkpoint
        except FileNotFoundError:
            raise  # this should never happen

    def _save_checkpoint(self, epoch: int) -> None:
        checkpoint_dir = self._output_dir / "checkpoints"
        checkpoint_file = checkpoint_dir / f"checkpoint_{epoch+1}.tar"

        checkpoint = Checkpoint(
            epoch=epoch,
            model=self._model.state_dict(),
            criterion=self._criterion.state_dict(),
            optimizer=self._optimizer.state_dict(),
            lr_scheduler=self._lr_scheduler.state_dict(),
            scaler=self._scaler.state_dict(),
            metrics_tracker=self._em_tracker.state_dict(),
            losses_tracker=self._el_tracker.state_dict(),
        )

        checkpoint.save(checkpoint_file)
        self._logger.info(f"Saved checkpoint at epoch {epoch + 1}.")

    def _save_model(self, epoch: int) -> None:
        best_epoch: dict[str, int]
        _, best_epoch = self._em_tracker.best_metric(True)  # type: ignore
        if best_epoch["IoU"] != epoch:
            return

        self._logger.info(f"Saving model at epoch {epoch + 1}.")
        model_dir = self._output_dir / "weights.pth"
        torch.save(self._model.state_dict(), model_dir)

    def _set_loaders(self) -> None:
        train_dataset = Dataset.from_config(
            self._config.dataset,
            split=Split.TRAIN,
            debug=self._config.debug,
        )
        eval_dataset = Dataset.from_config(
            self._config.dataset,
            split=Split.VALIDATION,
            debug=self._config.debug,
        )

        train_sampler = data.RandomSampler(train_dataset)
        eval_sampler = data.SequentialSampler(eval_dataset)

        train_batch_sampler = data.BatchSampler(
            train_sampler,
            batch_size=(
                self._config.batch_size // self._config.gradient_accumulation_steps
            ),
            drop_last=True,
        )

        eval_batch_sampler = data.BatchSampler(
            eval_sampler,
            batch_size=1,
            drop_last=False,
        )

        self._train_loader = DataLoader(
            train_dataset,  # type: ignore
            batch_sampler=train_batch_sampler,
            collate_fn=Dataset.batchify,
        )

        self._eval_loader = DataLoader(
            eval_dataset,  # type: ignore
            batch_sampler=eval_batch_sampler,
            collate_fn=Dataset.batchify,
        )

        self._logger.info(f"Using dataset {self._config.dataset.name}.")
        self._logger.info(
            f"\t(train) size: {len(train_dataset)} | (eval) size: {len(eval_dataset)}"
        )
        self._logger.info(
            f"\t(train) batch size: {self._config.batch_size} | (eval) batch size: 1"
        )
        self._logger.info(
            "\t(train) gradient accumulation steps: "
            f"{self._config.gradient_accumulation_steps}"
        )

    def _set_model(self, checkpoint: Optional[Checkpoint] = None) -> None:
        self._logger.info(f"Using model {self._config.model.name}.")

        model: Model[_T] = Model.from_config(self._config.model)
        criterion = model.criterion
        if criterion is None:
            raise ValueError(
                f"Model {self._config.model.name} cannot be trained "
                "since no criterion is defined."
            )

        model = model.to(self._device.to_torch())
        criterion = criterion.to(self._device.to_torch())

        if checkpoint is not None:
            model.load_state_dict(checkpoint.model)
            criterion.load_state_dict(checkpoint.criterion)
            self._logger.info("Loaded model weights from checkpoint.")

        if self._config.compile:
            self._logger.info("Compiling model.")
            model = torch.compile(model)  # type: ignore
        else:
            self._logger.info("Skipping model compilation.")

        self._model = model
        self._criterion = criterion

    def _set_optimizer(self, checkpoint: Optional[Checkpoint] = None) -> None:
        self._logger.info(f"Using optimizer {self._config.optimizer.name}.")

        optimizer = Optimizer.from_config(
            self._config.optimizer,
            [param for param in self._model.parameters() if param.requires_grad],
        )

        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint.optimizer)
            self._logger.info("Loaded optimizer state from checkpoint.")

        self._optimizer = optimizer

    def _set_lr_scheduler(self, checkpoint: Optional[Checkpoint] = None) -> None:
        self._logger.info(f"Using lr scheduler {self._config.lr_scheduler.name}.")

        lr_scheduler = LRScheduler.from_config(
            self._config.lr_scheduler,
            self._optimizer,
        )

        if checkpoint is not None:
            lr_scheduler.load_state_dict(checkpoint.lr_scheduler)
            self._logger.info("Loaded lr scheduler state from checkpoint.")

        self._lr_scheduler = lr_scheduler

    def _set_scaler(self, checkpoint: Optional[Checkpoint]) -> None:
        enabled = self._config.mixed_precision and self._device.is_cuda
        scaler = GradScaler(enabled=enabled)

        if checkpoint is not None:
            scaler.load_state_dict(checkpoint.scaler)

        self._scaler = scaler

    def _set_metrics(self) -> None:
        self._tl_tracker = LossTracker(self._criterion.losses).to(
            self._device.to_torch()
        )

        self._el_tracker = LossTracker(self._criterion.losses).to(
            self._device.to_torch()
        )

        metrics = tm.MetricCollection(
            {
                "IoU": IoU(),
                "GIoU": GIoU(),
            }
        )

        self._tm_tracker = tm.MetricTracker(
            metrics,
            maximize=True,
        ).to(self._device.to_torch())

        self._em_tracker = tm.MetricTracker(
            metrics.clone(),
            maximize=True,
        ).to(self._device.to_torch())

    def _log_statistics(self, elapsed: float, train: bool) -> None:
        self._logger.info("Statistics:")
        self._logger.info(f"\telapsed time: {elapsed:.2f} s.")
        num_batches = len(self._train_loader) if train else len(self._eval_loader)
        self._logger.info(f"\ttime per batch: {elapsed / num_batches:.2f} s.")

        if train:
            self._logger.info(f"\tlearning rate: {self._lr_scheduler.last_lr:.4f}")

        metrics_tracker = self._tm_tracker if train else self._em_tracker
        metrics = metrics_tracker.compute()

        self._logger.info("\tmetrics:")
        for key, value in metrics.items():
            self._logger.info(f"\t\t{key}: {value.item():.4f}")

        losses_tracker = self._tl_tracker if train else self._el_tracker
        losses = losses_tracker.compute()
        self._logger.info("\tlosses:")
        for key, value in losses.items():
            self._logger.info(f"\t\t{key}: {value.item():.4f}")

    def _train_epoch(self, epoch: int) -> None:
        self._logger.info(f"Training epoch {epoch + 1} started.")

        start = timer()

        self._model.train()
        self._tl_tracker.increment()
        self._tm_tracker.increment()
        self._optimizer.zero_grad()

        num_batches = (
            len(self._train_loader.dataset) // self._config.batch_size  # type: ignore
        )
        counter = tqdm(
            desc="Training",
            total=num_batches * self._config.batch_size,
        )

        batch: Batch
        bboxes: BBoxes
        for idx, (batch, bboxes) in enumerate(self._train_loader):
            # this is done since the batch size of the dataloader is equal to
            # batch_size / gradient_accumulation_steps
            # thus the samples belonging to the last non complete batch
            # will not be used for training, so we stop earlier
            if counter.total == counter.n:
                break

            batch = batch.to(self._device.to_torch())
            bboxes = bboxes.to(self._device.to_torch())

            device_type = "cuda" if self._device.is_cuda else "cpu"
            with autocast(device_type, enabled=self._config.mixed_precision):
                outputs = self._model(batch)
                tmp_losses = self._criterion(outputs, bboxes)
                losses = self._tl_tracker(tmp_losses)
                loss = losses["total_loss"] / self._config.gradient_accumulation_steps

            self._scaler.scale(loss).backward()

            if (idx + 1) % self._config.gradient_accumulation_steps == 0:
                if self._config.max_grad_norm is not None:
                    self._scaler.unscale_(self._optimizer)
                    torch.nn.utils.clip_grad_norm_(  # type: ignore
                        self._model.parameters(), self._config.max_grad_norm
                    )

                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad()
                self._lr_scheduler.step_after_batch()

            with torch.no_grad():
                pred_bboxes = self._model.predict(outputs)
                self._tm_tracker.update(pred_bboxes.xyxyn, bboxes.xyxyn)

            counter.update(len(batch))

        self._lr_scheduler.step_after_epoch()

        end = timer()
        elapsed = end - start

        self._logger.info(f"Training epoch {epoch + 1} finished.")
        self._log_statistics(elapsed, train=True)

    @torch.no_grad()
    def _eval_epoch(self, epoch: int) -> None:
        self._logger.info(f"Evaluating epoch {epoch + 1} started.")

        start = timer()

        self._model.eval()
        self._el_tracker.increment()
        self._em_tracker.increment()

        batch: Batch
        bboxes: BBoxes
        for batch, bboxes in tqdm(self._eval_loader, desc="Evaluating"):
            batch = batch.to(self._device.to_torch())
            bboxes = bboxes.to(self._device.to_torch())

            device_type = "cuda" if self._device.is_cuda else "cpu"
            with autocast(device_type, enabled=self._config.mixed_precision):
                outputs = self._model(batch)
                tmp_losses = self._criterion(outputs, bboxes)
                self._el_tracker.update(tmp_losses)

            pred_bboxes = self._model.predict(outputs)
            self._em_tracker.update(pred_bboxes.xyxyn, bboxes.xyxyn)

        end = timer()
        elapsed = end - start

        self._logger.info(f"Evaluating epoch {epoch + 1} finished.")
        self._log_statistics(elapsed, train=False)

    def _run(self, start_epoch: int) -> None:
        if start_epoch == self._config.num_epochs:
            self._logger.info("Training already finished.")
            return

        self._logger.info("Training started.")

        start = timer()

        for epoch in range(start_epoch, self._config.num_epochs):
            self._logger.info(f"Epoch {epoch + 1}/{self._config.num_epochs} started.")

            if self._device.is_cuda:
                torch.cuda.empty_cache()

            self._train_epoch(epoch)

            if self._device.is_cuda:
                torch.cuda.empty_cache()

            self._eval_epoch(epoch)

            self._logger.info(f"Epoch {epoch + 1}/{self._config.num_epochs} finished.")

            if (epoch + 1) % self._config.checkpoint_interval == 0:
                self._save_checkpoint(epoch)

            self._save_model(epoch)

        end = timer()
        elapsed = end - start

        self._logger.info("Training finished.")
        self._logger.info("Statistics:")
        self._logger.info(f"\telapsed time: {elapsed:.2f} s")
        num_epochs = self._config.num_epochs - start_epoch
        self._logger.info(f"\ttime per epoch: {elapsed / num_epochs:.2f} s")

        best_metrics: dict[str, float]
        best_epoch: dict[str, int]

        best_metrics, best_epoch = self._em_tracker.best_metric(True)  # type: ignore
        self._logger.info("\tbest metrics:")
        for key, value in best_metrics.items():
            self._logger.info(f"\t\t{key}: {value:.4f} | epoch: {best_epoch[key]}")

        best_losses, best_epoch = self._el_tracker.best_loss(True)  # type: ignore
        self._logger.info("\tbest losses:")
        for key, value in best_losses.items():
            self._logger.info(f"\t\t{key}: {value:.4f} | epoch: {best_epoch[key]}")

    def run(self) -> None:
        try:
            self._setup_environment()
            self._set_device()
            checkpoint = self._load_checkpoint()
            self._set_loaders()
            self._set_model(checkpoint)
            self._set_optimizer(checkpoint)
            self._set_lr_scheduler(checkpoint)
            self._set_scaler(checkpoint)
            self._set_metrics()

            start_epoch = 0
            if checkpoint is not None:
                start_epoch = checkpoint.epoch + 1
            del checkpoint

            self._run(start_epoch)

        except Exception as e:
            self._logger.error(f"Training failed with the following error: {e}")
            raise
