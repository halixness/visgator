##
##
##

import argparse
import dataclasses
from pathlib import Path
from typing import Any

from deepsight.engines.tester import Config, Tester


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", type=bool)
    return parser


def main() -> None:
    parser = get_arg_parser()
    args = parser.parse_args()

    config = Config.from_file(Path(args.config))
    if args.debug is not None:
        config = dataclasses.replace(config, debug=args.debug)

    trainer: Tester[Any, Any] = Tester.new(config)
    trainer.run()


if __name__ == "__main__":
    main()
