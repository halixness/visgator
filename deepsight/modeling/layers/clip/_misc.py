##
##
##

import enum

from typing_extensions import Self


class Models(enum.Enum):
    ViT_B_32_224 = "B32"
    ViT_B_16_224 = "B16"
    ViT_L_14_224 = "L14"
    Vit_L_14_336 = "L14_336"

    @classmethod
    def from_str(cls, model: str) -> Self:
        match model:
            case "B32":
                return cls.ViT_B_32_224  # type: ignore
            case "B16":
                return cls.ViT_B_16_224  # type: ignore
            case "L14":
                return cls.ViT_L_14_224  # type: ignore
            case "L14_336":
                return cls.Vit_L_14_336  # type: ignore
            case _:
                return cls[model.upper()]

    def weights(self) -> str:
        match self:
            case self.ViT_B_32_224:
                return "openai/clip-vit-base-patch32"
            case self.ViT_B_16_224:
                return "openai/clip-vit-base-patch16"
            case self.ViT_L_14_224:
                return "openai/clip-vit-large-patch32"
            case self.Vit_L_14_336:
                return "openai/clip-vit-large-patch32-336"
            case _:
                raise ValueError(f"Unknown model type: {self}")
