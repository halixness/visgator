##
##
##

import enum

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
    def __init__(self, type: DeviceType | str | None, id: int | None = None) -> None:
        if type is None:
            type = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(type, str):
            type = DeviceType.from_str(type)

        self._type = type
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

    @property
    def is_cuda(self) -> bool:
        return self._type == DeviceType.CUDA

    def to_torch(self) -> torch.device:
        return torch.device(str(self))

    def __str__(self) -> str:
        if self._type == DeviceType.CPU:
            return str(self._type)
        return f"{self._type}:{self._id}"
