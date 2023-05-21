##
##
##

import enum
from typing import Optional

import torch
from typing_extensions import Self


class DeviceType(enum.Enum):
    CPU = "cpu"
    CUDA = "cuda"

    @classmethod
    def from_str(cls, value: str) -> Self:
        return cls[value.upper().strip()]

    def __str__(self) -> str:
        return self.value


class Device:
    def __init__(self, type: str, id: Optional[int] = None) -> None:
        self._type = DeviceType.from_str(type)
        self._id = None
        if self._type == DeviceType.CUDA:
            if id is None:
                id = 0

            if id < 0:
                raise ValueError(f"Invalid CUDA device ID {id}. Must be >= 0.")
            if id >= torch.cuda.device_count():
                raise ValueError(
                    f"Invalid CUDA device ID {id}. "
                    f"Only {torch.cuda.device_count()} devices available."
                )

            self._id = id

    @classmethod
    def from_str(cls, value: str) -> Self:
        parts = value.split(":")
        if len(parts) == 1:
            return cls(parts[0])
        return cls(parts[0], int(parts[1]))

    @classmethod
    def default(cls) -> Self:
        if torch.cuda.is_available():
            return cls("cuda")
        return cls("cpu")

    @property
    def is_cuda(self) -> bool:
        return self._type == DeviceType.CUDA

    def to_torch(self) -> torch.device:
        return torch.device(str(self))

    def __str__(self) -> str:
        if self._type == DeviceType.CPU:
            return str(self._type)
        return f"{self._type}:{self._id}"
