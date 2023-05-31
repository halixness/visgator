##
##
##

import torchvision
from typing_extensions import Self

from visgator.datasets import Dataset as _Dataset
from visgator.datasets import Split
from visgator.utils.batch import BatchSample
from visgator.utils.bbox import BBox, BBoxFormat

from ._config import Config
from ._misc import get_preprocessed_samples, get_processed_samples


class Dataset(_Dataset):
    """RefCOCOg dataset."""

    def __init__(self, config: Config, split: Split, debug: bool) -> None:
        processed_path = config.path / f"annotations/info_{config.split_provider}.json"

        if processed_path.exists():
            samples = get_processed_samples(config, [split])[split]
        else:
            samples = get_preprocessed_samples(config, [split])[split]

        if debug:
            samples = samples[:100]

        self._samples = samples

    @classmethod
    def from_config(
        cls,
        config: Config,  # type: ignore
        split: Split,
        debug: bool = False,
    ) -> Self:
        return cls(config, split, debug)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[BatchSample, BBox]:
        sample = self._samples[index]

        image = torchvision.io.read_image(str(sample.path))
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        data = BatchSample(image, sample.caption)
        bbox = BBox(
            box=sample.bbox,
            image_size=image.shape[1:],
            format=BBoxFormat.XYWH,
            normalized=False,
        )

        return data, bbox
