##
##
##

import logging
import os

import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
from torchmetrics import MetricCollection
from tqdm import tqdm
from visgator.datasets import Dataset, Split
from visgator.metrics import JaccardIndex
from visgator.models import Model
from visgator.utils.batch import Batch
from visgator.utils.bbox import BBoxes

from ._config import Config, Device


class Evaluator:
    def __init__(self, config: Config) -> None:
        self._config = config

        self._logger: logging.Logger
        self._device: torch.device
        self._loader: DataLoader[tuple[Batch, BBoxes]]
        self._model: Model
        self._metrics: MetricCollection

    def _set_logger(self) -> None:
        self._logger = logging.getLogger(Evaluator.__class__.__name__)

        for handler in self._logger.handlers:
            self._logger.removeHandler(handler)
        for filter in self._logger.filters:
            self._logger.removeFilter(filter)

        if self._config.debug:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)

    def _init_enviroment(self) -> None:
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self._config.seed)
        torch.cuda.manual_seed(self._config.seed)
        torch.backends.cudnn.enabled = True

        if self._config.debug:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        else:
            torch.backends.cudnn.benchmark = True
            torch.use_deterministic_algorithms(False)  # type: ignore
            try:
                del os.environ["CUBLAS_WORKSPACE_CONFIG"]
                # just to be sure
                os.unsetenv("CUBLAS_WORKSPACE_CONFIG")
            except KeyError:
                pass

    def _set_device(self) -> None:
        if self._config.device == Device.CPU:
            self._device = torch.device("cpu")
            self._logger.info("Using CPU.")
        elif not torch.cuda.is_available():
            self._device = torch.device("cpu")
            self._logger.warning("CUDA is not available, using CPU.")
        else:
            self._device = torch.device("cuda")
            self._logger.info("Using CUDA.")

    def _set_dataloader(self) -> None:
        dataset = Dataset.from_config(self._config.dataset, split=Split.TEST)
        sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=1, drop_last=False)

        self._loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=1,
            pin_memory=True,
            collate_fn=dataset.batchify,
        )

        self._logger.info(f"Using dataset {self._config.dataset.name}:")
        self._logger.info(f"\tsize: {len(dataset)}")
        self._logger.info(f"\tbatch size: {1}")

    def _set_model(self) -> None:
        self._logger.info(f"Using model {self._config.model.name}.")

        model = Model.from_config(self._config.model)
        self._model = model.to(self._device)

        if self._config.weights is None:
            self._logger.warning("No weights loaded, using initialization weights.")
            return

        try:
            weights = torch.load(self._config.weights, map_location=self._device)
            self._model.load_state_dict(weights)
            self._logger.info("Loaded model weights.")
        except FileNotFoundError:
            self._logger.error(f"No weights found at '{self._config.weights}'.")
            raise

    def _set_metrics(self) -> None:
        self._metrics = MetricCollection(
            {
                "jaccard": JaccardIndex(),
            }
        ).to(self._device)

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
            batch = batch.to(self._device)
            bboxes = bboxes.to(self._device)

            output = self._model(batch)
            predictions = self._model.predict(output)

            self._metrics.update(predictions.xyxyn, bboxes.xyxyn)

        self._log_metrics()
        self._logger.info("Evaluation finished.")

    def run(self) -> None:
        try:
            self._set_logger()
            self._init_enviroment()
            self._set_device()
            self._set_dataloader()
            self._set_model()
            self._set_metrics()

            self._eval()

        except Exception as e:
            self._logger.error(f"Training failed with the following error: {e}")
            raise
