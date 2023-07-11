##
##
##

from pathlib import Path

from datasets.refcocog import Config as RefCOCOGConfig
from deepsight.engines.tester import Config
from deepsight.utils.wandb import Config as WandbConfig

from .models.sgg_owlvit_d3 import config as pipeline_config

config = Config(
    dataset=RefCOCOGConfig(Path("data/refcocog")),
    pipeline=pipeline_config,
    wandb=WandbConfig(
        enabled=True,
        job_type="test",
        project="tests",
        entity="visgator",
        save=False,
    ),
    weights=Path("weights/sgg.ptrom"),
)
