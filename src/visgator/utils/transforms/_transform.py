##
##
##

import abc
from dataclasses import dataclass
from typing import Optional, Union, overload

import torch
from jaxtyping import UInt8
from torch import Tensor

from visgator.utils.bbox import BBox


@dataclass(frozen=True, slots=True)
class Transform(abc.ABC):
    @abc.abstractproperty
    def p(self) -> float:
        ...

    @abc.abstractmethod
    def _apply(
        self, img: UInt8[Tensor, "C H W"], bbox: Optional[BBox] = None
    ) -> tuple[UInt8[Tensor, "C H W"], Optional[BBox]]:
        ...

    @overload
    def __call__(
        self,
        img: UInt8[Tensor, "C H W"],
    ) -> UInt8[Tensor, "C H W"]:
        ...

    @overload
    def __call__(
        self,
        img: UInt8[Tensor, "C H W"],
        bbox: BBox,
    ) -> tuple[UInt8[Tensor, "C H W"], BBox]:
        ...

    def __call__(
        self,
        img: UInt8[Tensor, "C H W"],
        bbox: Optional[BBox] = None,
    ) -> Union[UInt8[Tensor, "C H W"], tuple[UInt8[Tensor, "C H W"], BBox]]:
        if torch.rand(1) > self.p:
            return img if bbox is None else (img, bbox)

        if bbox is None:
            return self._apply(img)[0]

        return self._apply(img, bbox)  # type: ignore
