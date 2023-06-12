##
##
##

import json
import logging
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
from typing import Generic, Optional, TypeVar

import torch
import torchmetrics as tm
import wandb
from torch import autocast  # type: ignore
from torch.cuda.amp import GradScaler  # type: ignore
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Self

from visgator.datasets import Dataset, Split
from visgator.lr_schedulers import LRScheduler
from visgator.metrics import GIoU, IoU, IoUAccuracy, LossTracker
from visgator.models import Criterion, Model, PostProcessor
from visgator.optimizers import Optimizer
from visgator.utils.batch import Batch
from visgator.utils.bbox import BBoxes
from visgator.utils.misc import setup_logger
from visgator.utils.torch import Device, init_torch

from ._config import Config, Params
from ._misc import Checkpoint

_T = TypeVar("_T")


class Trainer(Generic[_T]):
    def __init__(self, config: Config) -> None:
        self._config = config
        self._params: Params

        # set in the following order
        self._dir: Path
        self._resumed: bool
        self._logger: logging.Logger

        self._device: Device
        self._train_loader: DataLoader[tuple[Batch, BBoxes]]
        self._eval_loader: DataLoader[tuple[Batch, BBoxes]]
        self._model: Model[_T]
        self._criterion: Criterion[_T]
        self._postprocessor: PostProcessor[_T]
        self._optimizer: Optimizer
        self._lr_scheduler: LRScheduler
        self._scaler: GradScaler

        self._tl_tracker: LossTracker
        self._el_tracker: LossTracker
        self._tm_tracker: tm.MetricTracker
        self._em_tracker: tm.MetricTracker

    @classmethod
    def from_config(cls, config: Config) -> Self:
        return cls(config)

    def _setup_launch(self) -> None:
        if self._config.wandb.enabled:
            args = self._config.wandb.args
            assert args is not None

            if args.id is not None:
                if (self._config.dir / "wandb").exists():
                    raise ValueError(
                        f"Directory '{self._config.dir}' already contains a run. "
                        "If you want to resume this run, do not specify the 'id' "
                        "parameter. Otherwise, change the 'dir' parameter."
                    )

                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self._dir = self._config.dir / now
                self._dir.mkdir(parents=True, exist_ok=False)

                wandb.init(
                    project=args.project,
                    entity=args.entity,
                    dir=self._dir,
                    id=args.id,
                    resume="must",
                )
                self._resumed = True

                assert wandb.run is not None
                self._params = Params.from_dict(wandb.run.config)
            elif (self._config.dir / "wandb").exists():
                # resume from previous run using wandb
                self._dir = self._config.dir
                wandb.init(
                    project=args.project,
                    entity=args.entity,
                    dir=self._dir,
                    resume=True,
                )
                self._resumed = True

                assert wandb.run is not None
                self._params = Params.from_dict(wandb.run.config)
            elif (self._config.dir / "config.json").exists():
                raise ValueError("Cannot resume with wandb a run started locally.")
            else:
                # start a new run using wandb
                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self._dir = self._config.dir / now
                self._dir.mkdir(parents=True, exist_ok=False)

                wandb.init(
                    project=args.project,
                    entity=args.entity,
                    job_type=args.job_type,
                    name=args.name,
                    tags=args.tags,
                    notes=args.notes,
                    dir=self._dir,
                )
                self._resumed = False

                self._params = Params.from_dict(self._config.params)
        elif (self._config.dir / "config.json").exists():
            # resume from previous run locally
            self._resumed = True
            self._dir = self._config.dir
            config_file = self._config.dir / "config.json"
            with open(config_file, "r") as f:
                self._params = Params.from_dict(json.load(f))

            wandb.init(mode="disabled")
        else:
            # start a new run locally
            self._resumed = False
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._dir = self._config.dir / now
            self._dir.mkdir(parents=True, exist_ok=False)
            self._params = Params.from_dict(self._config.params)

            wandb.init(mode="disabled")

    def _set_device(self) -> None:
        if self._params.device is None:
            self._device = Device.default()
        else:
            self._device = Device.from_str(self._params.device)

        if not self._resumed:
            self._logger.info(f"Using device {self._device}.")

    def _load_checkpoint(self) -> Optional[Checkpoint]:
        if not self._resumed:
            return None

        if wandb.run is not None:
            checkpoint_io = wandb.restore("checkpoint.tar")
            if checkpoint_io is None:
                return None

            checkpoint = Checkpoint.from_file(
                Path(checkpoint_io.name),
                self._device.to_torch(),
            )

            self._logger.info("Loaded checkpoint from wandb.")
        else:
            checkpoint_file = self._dir / "checkpoint.tar"
            if not checkpoint_file.exists():
                return None

            checkpoint = Checkpoint.from_file(checkpoint_file, self._device.to_torch())
            self._logger.info("Loaded checkpoint from local directory.")

        return checkpoint

    def _save_config(self) -> None:
        if self._resumed:
            return

        config_file = self._dir / "config.json"
        config = self._config.to_dict()
        config.update(self._params.to_dict())
        wandb.config.update(config)

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    def _save_checkpoint(self, epoch: int) -> None:
        checkpoint_file = self._dir / "checkpoint.tar"

        checkpoint = Checkpoint(
            epoch=epoch,
            model=self._model.state_dict(),
            criterion=self._criterion.state_dict(),
            postprocessor=self._postprocessor.state_dict(),
            optimizer=self._optimizer.state_dict(),
            lr_scheduler=self._lr_scheduler.state_dict(),
            scaler=self._scaler.state_dict(),
            metrics_tracker=self._em_tracker.state_dict(),
            losses_tracker=self._el_tracker.state_dict(),
        )

        save = self._config.wandb.args.save if self._config.wandb.args else False
        checkpoint.save(checkpoint_file, wandb_save=save)
        self._logger.info(f"Saved checkpoint at epoch {epoch + 1}.")

    def _save_model(self, epoch: int) -> None:
        best_values: dict[str, float]
        best_epoch: dict[str, int]
        best_values, best_epoch = self._em_tracker.best_metric(True)  # type: ignore
        if best_epoch["IoU"] != epoch:
            return

        iou = best_values["IoU"]
        self._logger.info(f"Saving model at epoch {epoch + 1}.")
        model_dir = self._dir / f"{self._model.name}_epoch-{epoch}_iou-{iou}.pt"
        torch.save(self._model.state_dict(), model_dir)

        save = self._config.wandb.args.save if self._config.wandb.args else False
        if save:
            artifact = wandb.Artifact(
                type="model",
                name=f"{self._model.name}",
                metadata={"epoch": epoch, "iou": iou},
            )
            artifact.add_file(model_dir)
            assert wandb.run is not None
            wandb.run.log_artifact(artifact)

    def _set_loaders(self) -> None:
        train_dataset = Dataset.from_config(
            self._params.dataset,
            split=Split.TRAIN,
            debug=self._config.debug,
        )
        eval_dataset = Dataset.from_config(
            self._params.dataset,
            split=Split.VALIDATION,
            debug=self._config.debug,
        )

        train_sampler = data.RandomSampler(train_dataset)
        eval_sampler = data.SequentialSampler(eval_dataset)

        train_batch_sampler = data.BatchSampler(
            train_sampler,
            batch_size=(
                self._params.train_batch_size
                // self._params.gradient_accumulation_steps
            ),
            drop_last=True,
        )

        eval_batch_sampler = data.BatchSampler(
            eval_sampler,
            batch_size=self._params.eval_batch_size,
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

        if not self._resumed:
            self._logger.info(f"Using {train_dataset.name} dataset.")
            self._logger.info(
                f"\t(train) size: {len(train_dataset)} | "
                f"(eval) size: {len(eval_dataset)}"
            )
            self._logger.info(
                f"\t(train) batch size: {self._params.train_batch_size} | "
                f"(eval) batch size: {self._params.eval_batch_size}"
            )
            self._logger.info(
                "\t(train) gradient accumulation steps: "
                f"{self._params.gradient_accumulation_steps}"
            )

    def _get_steps_per_epoch(self) -> int:
        return len(self._train_loader) // self._params.gradient_accumulation_steps

    def _set_model(self, checkpoint: Optional[Checkpoint] = None) -> None:
        model: Model[_T] = Model.from_config(self._params.model)
        criterion = model.criterion
        postprocessor = model.postprocessor
        if criterion is None:
            raise ValueError(
                f"Model {model.name} cannot be trained "
                "since no criterion is defined."
            )

        model = model.to(self._device.to_torch())
        criterion = criterion.to(self._device.to_torch())
        postprocessor = postprocessor.to(self._device.to_torch())

        if not self._resumed:
            self._logger.info(f"Using {model.name} model.")

        if checkpoint is not None:
            model.load_state_dict(checkpoint.model)
            criterion.load_state_dict(checkpoint.criterion)
            postprocessor.load_state_dict(checkpoint.postprocessor)
            self._logger.info("Loaded model toolchain state from checkpoint.")

        if self._params.compile:
            if not self._resumed:
                self._logger.info("Compiling model.")
            model = torch.compile(model)  # type: ignore
        elif not self._resumed:
            self._logger.info("Skipping model compilation.")

        self._model = model
        self._criterion = criterion
        self._postprocessor = postprocessor

    def _set_optimizer(self, checkpoint: Optional[Checkpoint] = None) -> None:
        optimizer = Optimizer.from_config(
            self._params.optimizer,
            [param for param in self._model.parameters() if param.requires_grad],
        )

        if not self._resumed:
            self._logger.info(f"Using {optimizer.name} optimizer.")

        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint.optimizer)
            self._logger.info("Loaded optimizer state from checkpoint.")

        self._optimizer = optimizer

    def _set_lr_scheduler(self, checkpoint: Optional[Checkpoint] = None) -> None:
        lr_scheduler = LRScheduler.from_config(
            self._params.lr_scheduler,
            self._optimizer,
            self._params.num_epochs,
            self._get_steps_per_epoch(),
        )

        if not self._resumed:
            self._logger.info(f"Using {lr_scheduler.name} lr scheduler.")

        if checkpoint is not None:
            lr_scheduler.load_state_dict(checkpoint.lr_scheduler)
            self._logger.info("Loaded lr scheduler state from checkpoint.")

        self._lr_scheduler = lr_scheduler

    def _set_scaler(self, checkpoint: Optional[Checkpoint]) -> None:
        enabled = self._params.mixed_precision and self._device.is_cuda
        scaler = GradScaler(enabled=enabled)

        if checkpoint is not None:
            scaler.load_state_dict(checkpoint.scaler)

        self._scaler = scaler

    def _set_metrics(self) -> None:
        tl_tracker = LossTracker(self._criterion.losses)
        el_tracker = LossTracker(self._criterion.losses)
        self._tl_tracker = tl_tracker.to(self._device.to_torch())
        self._el_tracker = el_tracker.to(self._device.to_torch())

        metrics = tm.MetricCollection(
            {
                "IoU": IoU(),
                "GIoU": GIoU(),
                "Accuracy@50": IoUAccuracy(0.5),
                "Accuracy@75": IoUAccuracy(0.75),
                "Accuracy@90": IoUAccuracy(0.9),
            },
            compute_groups=False,
        )
        maximize = [metric.higher_is_better for metric in metrics.values()]
        tm_tracker = tm.MetricTracker(metrics, maximize)  # type: ignore
        em_tracker = tm.MetricTracker(metrics.clone(), maximize)  # type: ignore
        self._tm_tracker = tm_tracker.to(self._device.to_torch())
        self._em_tracker = em_tracker.to(self._device.to_torch())

        assert wandb.run is not None
        wandb.run.define_metric("train/lr", summary="none")

        for name, loss in self._tl_tracker.items():
            for phase in ["train", "eval"]:
                wandb.run.define_metric(
                    f"{phase}/losses/{name}",
                    summary="max" if loss.maximize else "min",
                )

        for name, metric in metrics.items():
            for phase in ["train", "eval"]:
                wandb.run.define_metric(
                    f"{phase}/metrics/{name}",
                    summary="max" if metric.higher_is_better else "min",
                )

    def _log_statistics(self, epoch: int, elapsed: float, train: bool) -> None:
        self._logger.info("Statistics:")
        self._logger.info(f"\telapsed time: {elapsed:.2f} s.")
        if train:
            num_batches = self._params.train_batch_size
        else:
            num_batches = self._params.eval_batch_size
        self._logger.info(f"\ttime per batch: {elapsed / num_batches:.2f} s.")

        if train:
            phase = "train"
            self._logger.info(f"\tlearning rate: {self._lr_scheduler.last_lr:.4f}")
            wandb.log({"train/lr": self._lr_scheduler.last_lr}, step=epoch)
        else:
            phase = "eval"

        metrics_tracker = self._tm_tracker if train else self._em_tracker
        metrics = metrics_tracker.compute()
        self._logger.info("\tmetrics:")
        for key, value in metrics.items():
            self._logger.info(f"\t\t{key}: {value.item():.4f}")
            wandb.log({f"{phase}/metrics/{key}": value.item()}, step=epoch)

        losses_tracker = self._tl_tracker if train else self._el_tracker
        losses = losses_tracker.compute()
        self._logger.info("\tlosses:")
        for key, value in losses.items():
            self._logger.info(f"\t\t{key}: {value.item():.4f}")
            wandb.log({f"{phase}/losses/{key}": value.item()}, step=epoch)

    def _train_epoch(self, epoch: int) -> None:
        self._logger.info(f"Training epoch {epoch + 1} started.")

        start = timer()

        self._model.train()
        self._postprocessor.train()
        self._criterion.train()

        self._tl_tracker.increment()
        self._tm_tracker.increment()
        self._optimizer.zero_grad()

        counter = tqdm(
            desc="Training",
            total=self._get_steps_per_epoch() * self._params.train_batch_size,
        )

        with counter as progress_bar:
            batch: Batch
            bboxes: BBoxes
            device_type = "cuda" if self._device.is_cuda else "cpu"
            for idx, (batch, bboxes) in enumerate(self._train_loader):
                # since the dataloader batch size is equal to true batch size
                # // gradient accumulation steps, the last samples in the dataloader
                # may not be enough to fill the true batch size, so we break the loop
                # for example, if the true batch size is 8, the gradient accumulation
                # steps is 4 and the dataset size is 50, the last 2 samples will be
                # ignored
                if progress_bar.total == progress_bar.n:
                    break

                batch = batch.to(self._device.to_torch())
                bboxes = bboxes.to(self._device.to_torch())

                with autocast(device_type, enabled=self._params.mixed_precision):
                    outputs = self._model(batch)
                    tmp_losses = self._criterion(outputs, bboxes)
                    losses = self._tl_tracker(tmp_losses)
                    loss = losses.total / self._params.gradient_accumulation_steps

                self._scaler.scale(loss).backward()

                if (idx + 1) % self._params.gradient_accumulation_steps == 0:
                    if self._params.max_grad_norm is not None:
                        self._scaler.unscale_(self._optimizer)
                        torch.nn.utils.clip_grad_norm_(  # type: ignore
                            self._model.parameters(), self._params.max_grad_norm
                        )

                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                    self._optimizer.zero_grad()
                    self._lr_scheduler.step_after_batch()

                with torch.no_grad():
                    pred_bboxes = self._postprocessor(outputs)
                    self._tm_tracker.update(
                        pred_bboxes.to_xyxy().normalize().tensor,
                        bboxes.to_xyxy().normalize().tensor,
                    )

                progress_bar.update(len(batch))
                
                del batch
                del bboxes
                del outputs
                del loss

                # Empty cache at each sample
                if self._device.is_cuda:
                    torch.cuda.empty_cache()

        self._lr_scheduler.step_after_epoch()

        end = timer()
        elapsed = end - start

        self._logger.info(f"Training epoch {epoch + 1} finished.")
        self._log_statistics(epoch, elapsed, train=True)

    @torch.no_grad()
    def _eval_epoch(self, epoch: int) -> None:
        self._logger.info(f"Evaluating epoch {epoch + 1} started.")

        start = timer()

        self._model.eval()
        self._postprocessor.eval()
        self._criterion.eval()

        self._el_tracker.increment()
        self._em_tracker.increment()

        counter = tqdm(
            desc="Evaluating",
            total=len(self._eval_loader.dataset),  # type: ignore
        )

        with counter as progress_bar:
            batch: Batch
            bboxes: BBoxes
            device_type = "cuda" if self._device.is_cuda else "cpu"
            for batch, bboxes in self._eval_loader:
                batch = batch.to(self._device.to_torch())
                bboxes = bboxes.to(self._device.to_torch())

                with autocast(device_type, enabled=self._params.mixed_precision):
                    outputs = self._model(batch)
                    tmp_losses = self._criterion(outputs, bboxes)
                    self._el_tracker.update(tmp_losses)

                pred_bboxes = self._postprocessor(outputs)
                self._em_tracker.update(
                    pred_bboxes.to_xyxy().normalize().tensor,
                    bboxes.to_xyxy().normalize().tensor,
                )

                progress_bar.update(len(batch))

        end = timer()
        elapsed = end - start

        self._logger.info(f"Evaluating epoch {epoch + 1} finished.")
        self._log_statistics(epoch, elapsed, train=False)

    def _run(self, start_epoch: int) -> None:
        if start_epoch == self._params.num_epochs:
            self._logger.info("Training already finished.")
            return

        if not self._resumed:
            self._logger.info("Training started.")
        else:
            self._logger.info("Training resumed.")

        start = timer()

        for epoch in range(start_epoch, self._params.num_epochs):
            self._logger.info(f"Epoch {epoch + 1}/{self._params.num_epochs} started.")
            
            if self._device.is_cuda:
                torch.cuda.empty_cache()

            self._train_epoch(epoch)

            if (epoch + 1) % self._params.eval_interval == 0:
                if self._device.is_cuda:
                    torch.cuda.empty_cache()

                self._eval_epoch(epoch)
                self._save_model(epoch)

            self._logger.info(f"Epoch {epoch + 1}/{self._params.num_epochs} finished.")

            if (epoch + 1) % self._params.checkpoint_interval == 0:
                self._save_checkpoint(epoch)

        end = timer()
        elapsed = end - start

        self._logger.info("Training finished.")
        self._logger.info("Statistics:")
        self._logger.info(f"\telapsed time: {elapsed:.2f} s")
        num_epochs = self._params.num_epochs - start_epoch
        self._logger.info(f"\ttime per epoch: {elapsed / num_epochs:.2f} s")

        best_metrics: dict[str, float]
        best_epoch: dict[str, int]

        best_metrics, best_epoch = self._em_tracker.best_metric(True)  # type: ignore
        self._logger.info("\tbest metrics:")
        for key, value in best_metrics.items():
            self._logger.info(
                f"\t\t{key}: {value:.4f} | "
                f"epoch: {(best_epoch[key] + 1) * self._params.eval_interval}"
            )

        best_losses = self._el_tracker.best_loss()
        self._logger.info("\tbest losses:")
        for key, (value, epoch) in best_losses.items():
            self._logger.info(
                f"\t\t{key}: {value:.4f} | "
                f"epoch: {(epoch + 1) * self._params.eval_interval}"
            )

    def run(self) -> None:
        self._setup_launch()
        try:
            self._logger = setup_logger(self._dir / "train.log", self._config.debug)
            init_torch(self._params.seed, self._config.debug)
            self._save_config()
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

        wandb.finish()
