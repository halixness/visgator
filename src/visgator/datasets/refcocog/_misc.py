##
##
##

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import serde
from typing_extensions import Self

from visgator.datasets import Split
from visgator.utils.batch import Caption

from ._config import Config


@serde.serde(type_check=serde.Strict)
@dataclass(frozen=True)
class Sample:
    path: Path
    bbox: tuple[float, float, float, float]
    caption: Caption = serde.field(
        serializer=Caption.to_dict,
        deserializer=Caption.from_dict,
    )

    def to_dict(self) -> dict[str, Any]:
        return serde.to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return serde.from_dict(cls, data)


def _from_refcoco_split(split: str) -> Split:
    match split:
        case "train":
            return Split.TRAIN
        case "val":
            return Split.VALIDATION
        case "test":
            return Split.TEST
        case _:
            raise ValueError(f"Invalid split: {split}")


def _to_refcoco_split(split: Split) -> str:
    match split:
        case Split.TRAIN:
            return "train"
        case Split.VALIDATION:
            return "val"
        case Split.TEST:
            return "test"
        case _:
            raise ValueError(f"Invalid split: {split}")


def get_preprocessed_samples(
    config: Config, splits: list[Split]
) -> dict[Split, list[Sample]]:
    refs_path = config.path / f"annotations/refs({config.split_provider}).p"
    instances_path = config.path / "annotations/instances.json"
    images_path = config.path / "images"

    splits_str = [_to_refcoco_split(split) for split in splits]

    info: dict[str, Any] = {}
    with open(refs_path, "rb") as pf, open(instances_path, "r") as jf:
        refs = pickle.load(pf)
        instances = json.load(jf)

    images = {}
    for image in instances["images"]:
        images[image["id"]] = images_path / image["file_name"]

    for ref in refs:
        if ref["split"] not in splits_str:
            continue

        sentences = [sent["raw"] for sent in ref["sentences"]]
        if info.get(ref["ann_id"]) is not None:
            info[ref["ann_id"]]["sentences"].extend(sentences)
        else:
            info[ref["ann_id"]] = {
                "path": images[ref["image_id"]],
                "sentences": sentences,
                "split": _from_refcoco_split(ref["split"]),
            }

    for annotation in instances["annotations"]:
        if annotation["id"] in info:
            info[annotation["id"]]["bbox"] = annotation["bbox"]

    samples: dict[Split, list[Sample]] = {}
    for sample_info in info.values():
        split = sample_info["split"]
        path = sample_info["path"]
        bbox = sample_info["bbox"]

        for sent in sample_info["sentences"]:
            sample = Sample(
                path=path,
                caption=Caption(sent),
                bbox=bbox,
            )

            if len(Caption(sent).graph.entities) > 0 and len(Caption(sent).graph.relations) > 0:
                samples.setdefault(split, []).append(sample)

    return samples


def get_processed_samples(
    config: Config, splits: list[Split]
) -> dict[Split, list[Sample]]:
    images_path = config.path / "images"
    info_path = config.path / f"annotations/info_{config.split_provider}.json"

    with open(info_path, "r") as f:
        info = json.load(f)

    samples: dict[Split, list[Sample]] = {}
    for split_str, split_info in info.items():
        split = Split.from_str(split_str)
        if split not in splits:
            continue

        for sample_info in split_info:
            path = images_path / sample_info["image"]
            caption = Caption.from_dict(sample_info["caption"])
            bbox = sample_info["bbox"]

            sample = Sample(
                path=path,
                caption=caption,
                bbox=bbox,
            )

            # Remove any sample with no entities found (parsing problem)
            if len(caption.graph.entities) > 0 and len(caption.graph.relations) > 0:
                samples.setdefault(split, []).append(sample)
            
    return samples
