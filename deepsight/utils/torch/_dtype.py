##
##
##

import enum

import torch
from typing_extensions import Self


class FloatType(enum.Enum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    @classmethod
    def from_str(cls, value: str) -> Self:
        return cls[value.lower()]

    def is_mixed_precision(self) -> bool:
        match self:
            case FloatType.BFLOAT16:
                return True
            case FloatType.FLOAT16:
                return True
            case _:
                return False

    def to_torch_dtype(self) -> torch.dtype:
        match self:
            case FloatType.BFLOAT16:
                return torch.bfloat16
            case FloatType.FLOAT16:
                return torch.float16
            case FloatType.FLOAT32:
                return torch.float32
            case FloatType.FLOAT64:
                return torch.float64

    def __str__(self) -> str:
        return self.value
