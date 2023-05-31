##
##
##

from typing import Optional

from typing_extensions import Self

from visgator.utils.batch import Batch
from visgator.utils.bbox import BBoxes

from .._model import Model as _Model
from ._config import Config
from ._criterion import Criterion
from ._decoder import Decoder
from ._detector import Detector
from ._encoders import build_encoders


class Model(_Model[BBoxes]):
    def __init__(self, config: Config) -> None:
        self._criterion = Criterion()

        self._detector = Detector(config.detector)
        self._vision, self._text = build_encoders(config.encoders)
        self._decoder = Decoder(config.decoder)

    @classmethod
    def from_config(cls, config: Config) -> Self:  # type: ignore
        return cls(config)

    @property
    def criterion(self) -> Optional[Criterion]:
        return self._criterion

    def forward(self, batch: Batch) -> BBoxes:
        detections = self._detector(batch)
        img_embeddings = self._vision(batch)
        text_embeddings = self._text(batch)

        self._decoder(
            img_embeddings,
            [sample.caption for sample in batch],
            text_embeddings,
            detections,
        )

        raise NotImplementedError
