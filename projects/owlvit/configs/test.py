##
##
##

from pathlib import Path

from datasets.refcocog import Config as RefCOCOGConfig
from deepsight.engines.tester import Config
from deepsight.utils.wandb import Config as WandbConfig
from projects.owlvit.modeling import Config as PipelineConfig

config = Config(
    dataset=RefCOCOGConfig(Path("data/refcocog")),
    pipeline=PipelineConfig(),
    wandb=WandbConfig(
        enabled=True,
        job_type="test",
        project="tests",
        entity="visgator",
        save=False,
    ),
)
