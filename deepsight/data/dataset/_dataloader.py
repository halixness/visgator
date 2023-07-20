##
##
##

from typing import Generic, Iterator, TypeVar

from torch.utils.data import DataLoader as _DataLoader

from deepsight.data.structs import Batch
from deepsight.utils.protocols import Moveable

from ._dataset import Dataset

T = TypeVar("T", bound=Moveable)
U = TypeVar("U", bound=Moveable)


class DataLoader(Generic[T, U]):
    """A data loader for a dataset."""

    def __init__(
        self,
        dataset: Dataset[T, U],
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
    ) -> None:
        self._loader = _DataLoader(
            dataset=dataset,  # type: ignore
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
        )

    @property
    def dataset(self) -> Dataset[T, U]:
        return self._loader.dataset  # type: ignore

    def _collate_fn(self, batch: list[tuple[T, U]]) -> tuple[Batch[T], Batch[U]]:
        inputs, targets = zip(*batch)

        return Batch(inputs), Batch(targets)

    def __iter__(self) -> Iterator[tuple[Batch[T], Batch[U]]]:
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)
