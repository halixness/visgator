##
##
##

from pathlib import Path

from deepsight.modeling.layers.clip import Models
from projects.sgg.modeling import (
    Config,
    CriterionConfig,
    DecoderConfig,
    DetectorConfig,
    EncodersConfig,
    PreprocessorConfig,
)

config = Config(
    preprocessor=PreprocessorConfig(
        file=Path("data/refcocog/annotations/scene_graphs.json"),
        token="",
    ),
    encoders=EncodersConfig(
        output_dim=256,
        model=Models.ViT_B_32_224,
    ),
    detector=DetectorConfig(
        box_threshold=0.25,
        num_queries=4,
    ),
    decoder=DecoderConfig(
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
    ),
    criterion=CriterionConfig(
        l1_cost=5.0,
        giou_cost=2.0,
        similarity_cost=2.0,
        l1_weight=5.0,
        giou_weight=2.0,
        infonce_weight=1.0,
        auxiliary=True,
    ),
)
