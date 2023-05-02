##
##
##

import argparse
import json
from pathlib import Path

from .engines.evaluator import Config as EvaluatorConfig
from .engines.evaluator import Evaluator
from .engines.trainer import Config as TrainerConfig
from .engines.trainer import Trainer


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--phase", type=str, default="train", choices=["train", "eval"])
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
        case _:
            raise ValueError(f"Unknown config file extention: {extention}")

    if args.debug:
        cfg["debug"] = True

    if args.phase == "eval":
        config = EvaluatorConfig.from_dict(cfg)
        evaluator = Evaluator(config)
        evaluator.run()
    else:
        config = TrainerConfig.from_dict(cfg)
        trainer = Trainer(config)
        trainer.run()


if __name__ == "__main__":
    main()
