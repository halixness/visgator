##
##
##

from dataclasses import dataclass

from jaxtyping import Float, Int
from torch import Tensor

from deepsight.data.structs import BoundingBoxes, SceneGraph, TensorImage
from deepsight.utils.torch import Batched3DTensors, BatchedGraphs


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Data structure for model input."""

    # necessary for the model to know the original size of the images
    # and for OwlViT which needs PIL images
    images: list[TensorImage]
    features: Batched3DTensors
    captions: list[str]
    graphs: list[SceneGraph]


@dataclass(frozen=True, slots=True)
class TextEmbeddings:
    caption: Float[Tensor, "D"]  # noqa: F821
    entities: Float[Tensor, "N D"]
    relations: Float[Tensor, "M D"]


@dataclass(frozen=True, slots=True)
class ModelOutput:
    captions: Float[Tensor, "B D"]
    graphs: list[BatchedGraphs]  # for each decoder layer
    # padded bounding boxes
    boxes: list[BoundingBoxes]  # (B, N, D)
    padded_entities: Int[Tensor, "B N"]
