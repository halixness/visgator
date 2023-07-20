##
##
##

from pathlib import Path

from datasets.refcocog import Config as RefCOCOGConfig
from deepsight.engines.tester import Config
from deepsight.modeling.detectors import YOLOModel
from deepsight.utils.wandb import Config as WandbConfig
from projects.yoloclip.modeling import Config as PipelineConfig

config = Config(
    dataset=RefCOCOGConfig(Path("data/refcocog")),
    pipeline=PipelineConfig(yolo=YOLOModel.EXTRA),
    wandb=WandbConfig(
        enabled=True,
        job_type="test",
        project="tests",
        entity="visgator",
        save=False,
    ),
)
