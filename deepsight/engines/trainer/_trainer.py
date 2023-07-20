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
from torch.cuda.amp import GradScaler  # type: ignore
from tqdm import tqdm
from typing_extensions import Self

from deepsight.data.dataset import DataLoader, Dataset, Split
from deepsight.data.structs import BoundingBoxes, RECInput, RECOutput
from deepsight.lr_schedulers import LRScheduler
from deepsight.measures.losses import LossTracker
from deepsight.measures.metrics import BoxIoU, BoxIoUAccuracy, GeneralizedBoxIoU
from deepsight.modeling.pipeline import Pipeline
from deepsight.optimizers import Optimizer
from deepsight.utils import init_environment, setup_logger
from deepsight.utils.torch import Device

from ._config import Config, Params
from ._structs import Checkpoint

ModelInput = TypeVar("ModelInput")
ModelOutput = TypeVar("ModelOutput")


class Trainer(Generic[ModelInput, ModelOutput]):
    def __init__(self, config: Config) -> None:
        self._config = config

        # set in the following order
        self._dir: Path
        self._resumed: bool
        self._logger: logging.Logger

        self._device: Device
        self._train_loader: DataLoader[RECInput, RECOutput]
        self._eval_loader: DataLoader[RECInput, RECOutput]
        self._pipeline: Pipeline[RECInput, RECOutput, ModelInput, ModelOutput]
        self._optimizer: Optimizer
        self._lr_scheduler: LRScheduler
        self._scaler: GradScaler

        self._tl_tracker: LossTracker
        self._el_tracker: LossTracker
        self._tm_tracker: tm.MetricTracker
        self._em_tracker: tm.MetricTracker

    @classmethod
    def new(cls, config: Config) -> Self:
        return cls(config)

    def _setup_launch(self) -> None:
        wandb_cfg = self._config.wandb
        if wandb_cfg.enabled:
            if wandb_cfg.id is not None:
                if (self._config.dir / "wandb").exists():
                    raise ValueError(
                        f"Directory '{self._config.dir}' already contains a run. "
                        "If you want to resume this run, do not specify the `id` "
                        "parameter. Otherwise, change the `dir` parameter."
                    )

                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self._dir = self._config.dir / now
                self._dir.mkdir(parents=True, exist_ok=False)

                wandb.init(
                    project=wandb_cfg.project,
                    entity=wandb_cfg.entity,
                    dir=self._dir,
                    id=wandb_cfg.id,
                    resume="must",
                )
                self._resumed = True

                assert wandb.run is not None
                self._config.params = Params.from_dict(wandb.run.config)
            elif (self._config.dir / "wandb").exists():
                # resume from previous run using wandb
                self._dir = self._config.dir
                wandb.init(
                    project=wandb_cfg.project,
                    entity=wandb_cfg.entity,
                    dir=self._dir,
                    resume=True,
                )
                self._resumed = True

                assert wandb.run is not None
                self._config.params = Params.from_dict(wandb.run.config)
            elif (self._config.dir / "config.json").exists():
                raise ValueError("Cannot resume with wandb a run started locally.")
            else:
                # start a new run using wandb
                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self._dir = self._config.dir / now
                self._dir.mkdir(parents=True, exist_ok=False)

                wandb.init(
                    project=wandb_cfg.project,
                    entity=wandb_cfg.entity,
                    job_type=wandb_cfg.job_type,
                    name=wandb_cfg.name,
                    tags=wandb_cfg.tags,
                    notes=wandb_cfg.notes,
                    dir=self._dir,
                )
                self._resumed = False

                if self._config.params is None:
                    raise ValueError("Cannot start a new run without parameters.")

        elif (self._config.dir / "config.json").exists():
            # resume from previous run locally
            self._resumed = True
            self._dir = self._config.dir
            config_file = self._config.dir / "config.json"
            with config_file.open("r") as f:
                self._config.params = Params.from_dict(json.load(f))

            wandb.init(mode="disabled")
        else:
            # start a new run locally
            self._resumed = False
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._dir = self._config.dir / now
            self._dir.mkdir(parents=True, exist_ok=False)
            if self._config.params is None:
                raise ValueError("Cannot start a new run without parameters.")

            wandb.init(mode="disabled")

    def _set_device(self) -> None:
        assert self._config.params is not None
        self._device = Device(self._config.params.device)

        if not self._resumed:
            self._logger.info(f"Using device {self._device}.")

    def _load_checkpoint(self) -> Checkpoint | None:
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
        wandb.config.update(config)

        with config_file.open("w") as f:
            json.dump(config, f, indent=2)

    def _save_checkpoint(self, epoch: int) -> None:
        checkpoint_file = self._dir / "checkpoint.tar"

        checkpoint = Checkpoint(
            epoch=epoch,
            pipeline=self._pipeline.state_dict(),
            optimizer=self._optimizer.state_dict(),
            lr_scheduler=self._lr_scheduler.state_dict(),
            scaler=self._scaler.state_dict(),
            metrics_tracker=self._em_tracker.state_dict(),
            losses_tracker=self._el_tracker.state_dict(),
        )

        checkpoint.save(checkpoint_file, wandb_save=self._config.wandb.save)
        self._logger.info(f"Saved checkpoint at epoch {epoch + 1}.")

    def _save_model(self, epoch: int) -> None:
        best_values: dict[str, float]
        best_epoch: dict[str, int]
        best_values, best_epoch = self._em_tracker.best_metric(True)  # type: ignore
        if best_epoch["IoU"] != epoch:
            return

        iou = best_values["IoU"]
        self._logger.info(f"Saving model at epoch {epoch + 1}.")
        model_dir = self._dir / f"{self._pipeline.name}_epoch-{epoch}_iou-{iou}.pt"
        torch.save(self._pipeline.state_dict(), model_dir)

        if self._config.wandb.save:
            artifact = wandb.Artifact(
                type="pipeline",
                name=f"{self._pipeline.name}",
                metadata={"epoch": epoch, "iou": iou},
            )
            artifact.add_file(model_dir)
            assert wandb.run is not None
            wandb.run.log_artifact(artifact)

    def _set_loaders(self) -> None:
        """Sets the train and eval data loaders."""

        assert self._config.params is not None
        params = self._config.params

        train_dataset = Dataset.new_for_rec(
            params.dataset, split=Split.TRAIN, debug=self._config.debug
        )
        eval_dataset = Dataset.new_for_rec(
            params.dataset, split=Split.VALIDATION, debug=self._config.debug
        )

        self._train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=params.train_batch_size // params.gradient_accumulation_steps,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )

        self._eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=self._config.params.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )

        if not self._resumed:
            self._logger.info(f"Using {train_dataset.name} dataset.")
            self._logger.info(
                f"\t(train) size: {len(train_dataset)} | "
                f"(eval) size: {len(eval_dataset)}"
            )
            self._logger.info(
                f"\t(train) batch size: {params.train_batch_size} | "
                f"(eval) batch size: {params.eval_batch_size}"
            )
            self._logger.info(
                "\t(train) gradient accumulation steps: "
                f"{params.gradient_accumulation_steps}"
            )

    def _get_steps_per_epoch(self) -> int:
        assert self._config.params is not None
        params = self._config.params

        return len(self._train_loader) // params.gradient_accumulation_steps

    def _set_pipeline(self, checkpoint: Checkpoint | None) -> None:
        assert self._config.params is not None
        pipeline: Pipeline[RECInput, RECOutput, ModelInput, ModelOutput]

        pipeline = Pipeline.new_for_rec(self._config.params.pipeline)
        pipeline = pipeline.to(self._device.to_torch())

        if not self._resumed:
            self._logger.info(f"Using {pipeline.name} pipeline.")

        if checkpoint is not None:
            pipeline.load_state_dict(checkpoint.pipeline)
            self._logger.info("Loaded pipeline state from checkpoint.")

        self._pipeline = pipeline

    def _set_optimizer(self, checkpoint: Checkpoint | None) -> None:
        assert self._config.params is not None

        optimizer = Optimizer.new(
            self._config.params.optimizer, self._pipeline.named_parameters()
        )

        if not self._resumed:
            self._logger.info(f"Using {optimizer.name} optimizer.")

        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint.optimizer)
            self._logger.info("Loaded optimizer state from checkpoint.")

        self._optimizer = optimizer

    def _set_lr_scheduler(self, checkpoint: Checkpoint | None) -> None:
        assert self._config.params is not None
        params = self._config.params

        lr_scheduler = LRScheduler.new(
            params.lr_scheduler,
            self._optimizer,
            params.num_epochs,
            self._get_steps_per_epoch(),
        )

        if not self._resumed:
            self._logger.info(f"Using {lr_scheduler.name} lr scheduler.")

        if checkpoint is not None:
            lr_scheduler.load_state_dict(checkpoint.lr_scheduler)
            self._logger.info("Loaded lr scheduler state from checkpoint.")

        self._lr_scheduler = lr_scheduler

    def _set_scaler(self, checkpoint: Checkpoint | None) -> None:
        assert self._config.params is not None
        params = self._config.params

        enabled = params.dtype.is_mixed_precision() and self._device.is_cuda
        if params.init_scale is None:
            scaler = GradScaler(enabled=enabled)
        else:
            scaler = GradScaler(init_scale=params.init_scale, enabled=enabled)

        if checkpoint is not None:
            scaler.load_state_dict(checkpoint.scaler)

        self._scaler = scaler

    def _set_metrics(self) -> None:
        tl_tracker = LossTracker(self._pipeline.criterion.losses_names())
        el_tracker = LossTracker(self._pipeline.criterion.losses_names())
        self._tl_tracker = tl_tracker.to(self._device.to_torch())
        self._el_tracker = el_tracker.to(self._device.to_torch())

        metrics = tm.MetricCollection(
            {
                "IoU": BoxIoU(),
                "GIoU": GeneralizedBoxIoU(),
                "Accuracy@50": BoxIoUAccuracy(0.5),
                "Accuracy@75": BoxIoUAccuracy(0.75),
                "Accuracy@90": BoxIoUAccuracy(0.9),
            },
            compute_groups=False,
        )
        maximize = [metric.higher_is_better for metric in metrics.values()]
        tm_tracker = tm.MetricTracker(metrics, maximize)  # type: ignore
        em_tracker = tm.MetricTracker(metrics.clone(), maximize)  # type: ignore
        self._tm_tracker = tm_tracker.to(self._device.to_torch())
        self._em_tracker = em_tracker.to(self._device.to_torch())

        assert wandb.run is not None
        for name in self._optimizer.get_param_groups_names():
            wandb.run.define_metric(f"train/lr/{name}", summary="none")

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
        assert self._config.params is not None
        params = self._config.params

        self._logger.info("Statistics:")
        self._logger.info(f"\telapsed time: {elapsed:.2f} s.")
        if train:
            num_batches = params.train_batch_size
        else:
            num_batches = params.eval_batch_size
        self._logger.info(f"\ttime per batch: {elapsed / num_batches:.2f} s.")

        if train:
            phase = "train"
            lrs = {}
            self._logger.info("\tlearning rate:")
            for name, lr in zip(
                self._optimizer.get_param_groups_names(),
                self._lr_scheduler.get_last_lr(),
            ):
                self._logger.info(f"\t\t{name}: {lr:.10f}")
                lrs[f"train/lr/{name}"] = lr

            wandb.log(lrs, step=epoch)
        else:
            phase = "eval"

        metrics_tracker = self._tm_tracker if train else self._em_tracker
        metrics = metrics_tracker.compute()
        self._logger.info("\tmetrics:")
        for key, value in metrics.items():
            self._logger.info(f"\t\t{key}: {value.item():.5f}")
            wandb.log({f"{phase}/metrics/{key}": value.item()}, step=epoch)

        losses_tracker = self._tl_tracker if train else self._el_tracker
        losses = losses_tracker.compute()
        self._logger.info("\tlosses:")
        for key, value in losses.flatten():
            self._logger.info(f"\t\t{key}: {value.item():.5f}")
            wandb.log({f"{phase}/losses/{key}": value.item()}, step=epoch)

    def _train_epoch(self, epoch: int) -> None:
        assert self._config.params is not None
        params = self._config.params

        self._logger.info(f"Training epoch {epoch + 1} started.")

        start = timer()

        self._pipeline.train()
        self._tl_tracker.increment()
        self._tm_tracker.increment()
        self._optimizer.zero_grad()

        counter = tqdm(
            desc="Training",
            total=self._get_steps_per_epoch() * params.train_batch_size,
        )

        with counter as progress_bar:
            device_type = "cuda" if self._device.is_cuda else "cpu"
            for idx, (inputs, outputs) in enumerate(self._train_loader):
                # since the dataloader batch size is equal to true batch size
                # // gradient accumulation steps, the last samples in the dataloader
                # may not be enough to fill the true batch size, so we break the loop
                # for example, if the true batch size is 8, the gradient accumulation
                # steps is 4 and the dataset size is 50, the last 2 samples will be
                # ignored
                if progress_bar.total == progress_bar.n:
                    break

                inputs = inputs.to(self._device.to_torch())
                outputs = outputs.to(self._device.to_torch())

                with autocast(
                    device_type,
                    enabled=params.dtype.is_mixed_precision(),
                    dtype=params.dtype.to_torch_dtype(),
                ):
                    model_input = self._pipeline.preprocessor(inputs, outputs)
                    model_output = self._pipeline.model(model_input)
                    crt_losses = self._pipeline.criterion(model_output, outputs)

                    losses = self._tl_tracker(crt_losses)
                    loss = losses.total / params.gradient_accumulation_steps

                self._scaler.scale(loss).backward()

                if (idx + 1) % params.gradient_accumulation_steps == 0:
                    if params.max_grad_norm is not None:
                        self._scaler.unscale_(self._optimizer)
                        torch.nn.utils.clip_grad_norm_(  # type: ignore
                            self._pipeline.parameters(), params.max_grad_norm
                        )

                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                    self._optimizer.zero_grad()
                    self._lr_scheduler.step_after_batch()

                with torch.no_grad():
                    predictions = self._pipeline.postprocessor(model_output)
                    pred_boxes = BoundingBoxes.stack(
                        [prediction.box for prediction in predictions]
                    )
                    tgt_boxes = BoundingBoxes.stack([output.box for output in outputs])
                    self._tm_tracker.update(pred_boxes, tgt_boxes)

                progress_bar.update(len(inputs))

        self._lr_scheduler.step_after_epoch()

        end = timer()
        elapsed = end - start

        self._logger.info(f"Training epoch {epoch + 1} finished.")
        self._log_statistics(epoch, elapsed, train=True)

    @torch.no_grad()
    def _eval_epoch(self, epoch: int) -> None:
        assert self._config.params is not None
        params = self._config.params

        self._logger.info(f"Evaluating epoch {epoch + 1} started.")

        start = timer()

        self._pipeline.eval()
        self._el_tracker.increment()
        self._em_tracker.increment()

        counter = tqdm(desc="Evaluating", total=len(self._eval_loader.dataset))

        with counter as progress_bar:
            device_type = "cuda" if self._device.is_cuda else "cpu"
            for inputs, outputs in self._eval_loader:
                inputs = inputs.to(self._device.to_torch())
                outputs = outputs.to(self._device.to_torch())

                with autocast(
                    device_type,
                    enabled=params.dtype.is_mixed_precision(),
                    dtype=params.dtype.to_torch_dtype(),
                ):
                    model_input = self._pipeline.preprocessor(inputs, None)
                    model_output = self._pipeline.model(model_input)
                    crt_losses = self._pipeline.criterion(model_output, outputs)
                    self._el_tracker.update(crt_losses)

                predictions = self._pipeline.postprocessor(model_output)
                pred_boxes = BoundingBoxes.stack(
                    [prediction.box for prediction in predictions]
                )
                tgt_boxes = BoundingBoxes.stack([output.box for output in outputs])
                self._em_tracker.update(pred_boxes, tgt_boxes)

                progress_bar.update(len(inputs))

        end = timer()
        elapsed = end - start

        self._logger.info(f"Evaluating epoch {epoch + 1} finished.")
        self._log_statistics(epoch, elapsed, train=False)

    def _run(self, start_epoch: int) -> None:
        assert self._config.params is not None
        params = self._config.params

        if start_epoch == params.num_epochs:
            self._logger.info("Training already finished.")
            return

        if not self._resumed:
            self._logger.info("Training started.")
        else:
            self._logger.info("Training resumed.")

        start = timer()

        for epoch in range(start_epoch, params.num_epochs):
            self._logger.info(f"Epoch {epoch + 1}/{params.num_epochs} started.")

            self._train_epoch(epoch)

            if self._device.is_cuda:
                torch.cuda.empty_cache()

            if (epoch + 1) % params.eval_interval == 0:
                if self._device.is_cuda:
                    torch.cuda.empty_cache()

                self._eval_epoch(epoch)
                self._save_model(epoch)

            self._logger.info(f"Epoch {epoch + 1}/{params.num_epochs} finished.")

            if (epoch + 1) % params.checkpoint_interval == 0:
                self._save_checkpoint(epoch)

        end = timer()
        elapsed = end - start

        self._logger.info("Training finished.")
        self._logger.info("Statistics:")
        self._logger.info(f"\telapsed time: {elapsed:.2f} s")
        num_epochs = params.num_epochs - start_epoch
        self._logger.info(f"\ttime per epoch: {elapsed / num_epochs:.2f} s")

        best_metrics: dict[str, float]
        best_epoch: dict[str, int]

        best_metrics, best_epoch = self._em_tracker.best_metric(True)  # type: ignore
        self._logger.info("\tbest metrics:")
        for key, value in best_metrics.items():
            self._logger.info(
                f"\t\t{key}: {value:.4f} | "
                f"epoch: {(best_epoch[key] + 1) * params.eval_interval}"
            )

        best_losses = self._el_tracker.best_loss()
        self._logger.info("\tbest losses:")
        for key, (value, epoch) in best_losses.flatten():
            self._logger.info(
                f"\t\t{key}: {value:.4f} | "
                f"epoch: {(epoch + 1) * params.eval_interval}"
            )

    def run(self) -> None:
        self._setup_launch()
        try:
            assert self._config.params is not None
            self._logger = setup_logger(self._dir / "train.log", self._config.debug)
            init_environment(self._config.params.seed, self._config.debug)
            # TODO: solve issues with pyserde serialization
            # self._save_config()
            self._set_device()

            checkpoint = self._load_checkpoint()
            self._set_loaders()
            self._set_pipeline(checkpoint)
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
