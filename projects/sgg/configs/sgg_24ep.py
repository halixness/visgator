##
##
##

from pathlib import Path

from datasets.refcocog import Config as RefCOCOGConfig
from deepsight.engines.trainer import Config, Params
from deepsight.lr_schedulers.torch import Config as LRSchedulerConfig
from deepsight.optimizers.torch import Config as OptimizerConfig
from deepsight.optimizers.torch import ParamGroupConfig
from deepsight.utils import wandb
from deepsight.utils.torch import FloatType

from .models.sgg_owlvit_d3 import config as pipeline_config

config = Config(
    dir=Path("output/sgg"),
    wandb=wandb.Config(
        job_type="train",
        enabled=False,
        project="sgg",
        entity="visgator",
    ),
    debug=False,
    params=Params(
        num_epochs=24,
        train_batch_size=256,
        eval_batch_size=16,
        gradient_accumulation_steps=16,
        max_grad_norm=5.0,
        init_scale=2**12,
        seed=3407,
        dtype=FloatType.BFLOAT16,
        dataset=RefCOCOGConfig(path=Path("data/refcocog")),
        pipeline=pipeline_config,
        optimizer=OptimizerConfig(
            "AdamW",
            groups=[
                ParamGroupConfig(
                    regex=r"model.vision_encoder.projection.*",
                    args={"lr": 5e-4, "weight_decay": 1e-4},
                ),
                ParamGroupConfig(
                    regex=r"model.text_encoder.projection.*",
                    args={"lr": 5e-4, "weight_decay": 1e-4},
                ),
                ParamGroupConfig(
                    regex=r"model.vision_encoder.*",
                    args={"lr": 2e-4, "weight_decay": 0.0},
                ),
                ParamGroupConfig(
                    regex=r"model.text_encoder.*",
                    args={"lr": 2e-6, "weight_decay": 0.0},
                ),
                ParamGroupConfig(
                    regex=r"model.*", args={"lr": 5e-4, "weight_decay": 1e-4}
                ),
            ],
            freeze=["model.detector.*"],
        ),
        lr_scheduler=LRSchedulerConfig(
            "OneCycleLR", args={"max_lr": [5e-4, 5e-4, 2e-4, 2e-6, 5e-4]}
        ),
    ),
)
