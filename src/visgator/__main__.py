##
##
##

import argparse
import json
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from visgator.datasets import Config as DatasetConfig
from visgator.datasets import Generator
from visgator.engines.evaluator import Config as EvaluatorConfig
from visgator.engines.evaluator import Evaluator
from visgator.engines.trainer import Config as TrainerConfig
from visgator.engines.trainer import Trainer


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--phase", type=str, default="train", choices=["train", "eval", "generate"]
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    parser = get_arg_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    extention = config_path.suffix
    match extention:
        case ".json":
            with open(config_path, "r") as f:
                cfg = json.load(f)
        case ".yaml":
            yaml = YAML(typ="safe")
            cfg = yaml.load(config_path)
        case _:
            raise ValueError(f"Unknown config file extention: {extention}.")

    if args.debug:
        cfg["debug"] = True

    match args.phase:
        case "train":
            train_config = TrainerConfig.from_dict(cfg)
            trainer: Trainer[Any] = Trainer(train_config)
            trainer.run()
        case "eval":
            eval_config = EvaluatorConfig.from_dict(cfg)
            evaluator: Evaluator[Any] = Evaluator(eval_config)
            evaluator.run()
        case "generate":
            dataset_config = DatasetConfig.from_dict(cfg["dataset"])
            generator = Generator.new(dataset_config)
            generator.generate()
        case _:
            raise ValueError(f"Unknown phase: {args.phase}.")


if __name__ == "__main__":
    main()
