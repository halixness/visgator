##
##
##

from __future__ import annotations

import abc
import enum
import importlib
from typing import Generic, TypeVar

from typing_extensions import Self

from deepsight.data.structs import RECInput, RECOutput

from ._config import Config

T = TypeVar("T")
U = TypeVar("U")


class Split(enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

    @classmethod
    def from_str(cls, value: str) -> Self:
        """Deserializes a split enum from a string.

        Parameters
        ----------
        value : str
            The serialized split.

        Returns
        -------
        Self
            The deserialized split.
        """

        return cls[value.upper()]

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"Split.{self.name}"


class Dataset(abc.ABC, Generic[T, U]):
    """Abstract base class for all datasets."""

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The dataset name.

        Returns
        -------
        str
            The dataset name.
        """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    @classmethod
    @abc.abstractmethod
    def new_for_rec(
        cls, config: Config, split: Split, debug: bool
    ) -> Dataset[RECInput, RECOutput]:
        """Creates a new dataset given a `config` and `split` and `debug` flag
        for the Referring Expression Comprehension task.

        Calling this method on a subclass of `Dataset` will return an instance of
        that subclass. Calling it on `Dataset` will return an instance of a subclass
        of `Dataset` depending on the `module` field of the `config`. The `module`
        field should be a valid module path to a subclass of `Dataset`.

        Parameters
        ----------
        config : Config
            The dataset config.
        split : Split
            The dataset split.
        debug : bool
            Whether to run in debug mode.

        Returns
        -------
        Dataset[RECInput, RECOutput]
            The new dataset.
        """

        module = importlib.import_module(config.module)
        sub_cls = getattr(module, cls.__name__)
        return sub_cls.new_for_rec(  # type: ignore
            config=config,
            split=split,
            debug=debug,
        )

    # -------------------------------------------------------------------------
    # Magic methods
    # -------------------------------------------------------------------------

    @abc.abstractmethod
    def __getitem__(self, index: int) -> tuple[T, U]:
        """Gets the dataset sample at the given index.

        Parameters
        ----------
        index : int
            The sample index.

        Returns
        -------
        tuple[Input, Target]
            The input and target of the sample.
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Gets the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
