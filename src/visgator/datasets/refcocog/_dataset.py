##
##
##


import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torchvision

from visgator.utils.batch import BatchSample
from visgator.utils.bbox import BBox, BBoxFormat

from .._dataset import Dataset as BaseDataset
from .._dataset import Split
from ._config import Config


@dataclass(frozen=True)
class Sample:
    path: Path
    sentence: str
    bbox: tuple[float, float, float, float]


class Dataset(BaseDataset):
    """RefCOCOg dataset."""

    def __init__(self, config: Config, split: Split, debug: bool) -> None:
        super().__init__(config, split, debug)

        samples = self._get_samples(config, split)
        if debug:
            samples = samples[:100]

        self._samples = samples

    def _get_samples(self, config: Config, split: Split) -> list[Sample]:
        refs_path = config.path / f"annotations/refs({config.split_provider}).p"
        instances_path = config.path / "annotations/instances.json"
        images_path = config.path / "images"

        info: dict[str, Any] = {}
        with open(refs_path, "rb") as pf, open(instances_path, "r") as jf:
            refs = pickle.load(pf)
            instances = json.load(jf)

        images = {}
        for image in instances["images"]:
            images[image["id"]] = images_path / image["file_name"]

        for ref in refs:
            if ref["split"] != str(split):
                continue

            sentences = [sent["raw"] for sent in ref["sentences"]]
            if info.get(ref["ann_id"]) is not None:
                info[ref["ann_id"]]["sentences"].extend(sentences)
            else:
                info[ref["ann_id"]] = {
                    "path": images[ref["image_id"]],
                    "sentences": sentences,
                }

        for annotation in instances["annotations"]:
            if annotation["id"] in info:
                info[annotation["id"]]["bbox"] = annotation["bbox"]

        samples = []

        for sample_info in info.values():
            path = sample_info["path"]
            bbox = sample_info["bbox"]
            for sent in sample_info["sentences"]:
                sample = Sample(path, sent, bbox)
                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[BatchSample, BBox]:
        sample = self._samples[index]

        image = torchvision.io.read_image(str(sample.path))
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        data = BatchSample(image, sample.sentence)
        bbox = BBox.from_tuple(sample.bbox, image.shape[1:], BBoxFormat.XYWH)

        return data, bbox
