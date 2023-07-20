##
##
##

import json
import pickle
from pathlib import Path
from typing import Any

import torch
from typing_extensions import Self

from deepsight.data.dataset import Dataset as _Dataset
from deepsight.data.dataset import Split
from deepsight.data.structs import (
    BoundingBoxes,
    BoundingBoxFormat,
    RECInput,
    RECOutput,
    TensorImage,
)

from ._config import Config
from ._structs import Sample


class Dataset(_Dataset[RECInput, RECOutput]):
    """RefCOCOG dataset.

    .. note::
        At the moment, only the umd split is supported.
    """

    INVALID_SENT_IDS = set(
        [52285, 58412, 67725, 91442, 94554, 103508, 20655, 20656, 25625]
    )

    @property
    def name(self) -> str:
        return "RefCOCOg"

    def __init__(self, config: Config, split: Split, debug: bool) -> None:
        super().__init__()

        self._samples = get_samples(config.path, split)
        if debug:
            self._samples = self._samples[:100]

    @classmethod
    def new_for_rec(
        cls,
        config: Config,  # type: ignore
        split: Split,
        debug: bool,
    ) -> Self:
        return cls(config, split, debug)

    # --------------------------------------------------------------------------
    # Magic methods
    # --------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[RECInput, RECOutput]:
        sample = self._samples[index]

        image = TensorImage.open(sample.path)

        box = BoundingBoxes(
            torch.tensor(sample.bbox),
            images_size=image.size,
            format=BoundingBoxFormat.XYWH,
            normalized=False,
        )

        return RECInput(image, sample.description), RECOutput(box)


def get_samples(path: Path, split: Split) -> list[Sample]:
    refs_path = path / "annotations/refs(umd).p"
    instances_path = path / "annotations/instances.json"
    images_path = path / "images"

    info: dict[str, Any] = {}
    with open(refs_path, "rb") as pf, open(instances_path, "r") as jf:
        refs = pickle.load(pf)
        instances = json.load(jf)

    images = {}
    for image in instances["images"]:
        images[image["id"]] = images_path / image["file_name"]

    for ref in refs:
        if from_refcoco_split(ref["split"]) != split:
            continue

        sentences = [
            sent["raw"]
            for sent in ref["sentences"]
            if sent["sent_id"] not in Dataset.INVALID_SENT_IDS
        ]
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

    samples: list[Sample] = []
    for sample_info in info.values():
        path = sample_info["path"]
        bbox = sample_info["bbox"]

        for sent in sample_info["sentences"]:
            sample = Sample(
                path=path, bbox=tuple(bbox), description=sent  # type: ignore
            )

            samples.append(sample)

    return samples


def from_refcoco_split(split: str) -> Split:
    match split:
        case "train":
            return Split.TRAIN
        case "val":
            return Split.VALIDATION
        case "test":
            return Split.TEST
        case _:
            raise ValueError(f"Invalid split: {split}")
