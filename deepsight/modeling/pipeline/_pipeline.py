##
##
##

from __future__ import annotations

import abc
import importlib
from dataclasses import dataclass
from typing import Any, Generic, Iterable, TypeVar

import torch
from torch.nn import Parameter
from typing_extensions import Self

from deepsight.data.structs import RECInput, RECOutput
from deepsight.utils.protocols import Moveable

from ._config import Config
from ._criterion import Criterion
from ._model import Model
from ._postprocessor import PostProcessor
from ._preprocessor import PreProcessor

Input = TypeVar("Input", bound=Moveable)
Output = TypeVar("Output", bound=Moveable)
ModelInput = TypeVar("ModelInput")
ModelOutput = TypeVar("ModelOutput")


@dataclass(frozen=True, slots=True)
class Pipeline(abc.ABC, Moveable, Generic[Input, Output, ModelInput, ModelOutput]):
    """A pipeline is a sequence of steps that are executed in order to perform a
    specific task."""

    name: str
    preprocessor: PreProcessor[Input, Output, ModelInput]
    model: Model[ModelInput, ModelOutput]
    postprocessor: PostProcessor[ModelOutput, Output]
    criterion: Criterion[ModelOutput, Output]

    @classmethod
    @abc.abstractmethod
    def new_for_rec(
        cls, config: Config
    ) -> Pipeline[RECInput, RECOutput, ModelInput, ModelOutput]:
        """Creates a new pipeline given a `config` for the Referring Expression
        Comprehension task.

        Calling this method on a subclass of `Pipeline` will return an instance of
        that subclass. Calling it on `Pipeline` will return an instance of a subclass
        of `Pipeline` depending on the `module` field of the `config`. The `module`
        field should be a valid module path to a subclass of `Pipeline`.

        Parameters
        ----------
        config : Config
            The configuration for the pipeline.

        Returns
        -------
        Pipeline[RECInput, RECOutput, ModelInput, ModelOutput]
            The pipeline.
        """

        module = importlib.import_module(config.module)
        sub_cls = getattr(module, cls.__name__)
        return sub_cls.new_for_rec(config)  # type: ignore

    def state_dict(self) -> dict[str, dict[str, Any]]:
        """Returns the state of the pipeline as a dict.

        It contains the state of the preprocessor, model, postprocessor and criterion.

        Returns
        -------
        dict[str, dict[str, Any]]
            The state of the pipeline.
        """

        return {
            "preprocessor": self.preprocessor.state_dict(),
            "model": self.model.state_dict(),
            "postprocessor": self.postprocessor.state_dict(),
            "criterion": self.criterion.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, dict[str, Any]]) -> None:
        """Loads the pipeline state from the given `state_dict`.

        Parameters
        ----------
        state_dict : dict[str, dict[str, Any]]
            The state of the pipeline.
        """

        self.preprocessor.load_state_dict(state_dict["preprocessor"])
        self.model.load_state_dict(state_dict["model"])
        self.postprocessor.load_state_dict(state_dict["postprocessor"])
        self.criterion.load_state_dict(state_dict["criterion"])

    def to(self, device: torch.device | str) -> Self:
        """Moves the pipeline to the given `device`.

        Parameters
        ----------
        device : torch.device | str
            The device to move the pipeline to.

        Returns
        -------
        Self
            The pipeline moved to the given `device`.
        """

        return self.__class__(
            name=self.name,
            preprocessor=self.preprocessor.to(device),
            model=self.model.to(device),
            postprocessor=self.postprocessor.to(device),
            criterion=self.criterion.to(device),
        )

    def train(self) -> None:
        """Sets the whole pipeline to training mode."""

        self.preprocessor.train()
        self.model.train()
        self.postprocessor.train()
        self.criterion.train()

    def eval(self) -> None:
        """Sets the whole pipeline to evaluation mode."""

        self.preprocessor.eval()
        self.model.eval()
        self.postprocessor.eval()
        self.criterion.eval()

    def parameters(self) -> Iterable[Parameter]:
        """Returns an iterator over the pipeline parameters.

        Returns
        -------
        Iterable[Parameter]
            An iterator over the pipeline parameters.
        """

        yield from self.preprocessor.parameters()
        yield from self.model.parameters()
        yield from self.postprocessor.parameters()
        yield from self.criterion.parameters()

    def named_parameters(self) -> Iterable[tuple[str, Parameter]]:
        """Returns an iterator over the pipeline named parameters.

        Returns
        -------
        Iterable[tuple[str, Parameter]]
            An iterator over the pipeline named parameters.
        """

        for name, param in self.preprocessor.named_parameters():
            yield f"preprocessor.{name}", param

        for name, param in self.model.named_parameters():
            yield f"model.{name}", param

        for name, param in self.postprocessor.named_parameters():
            yield f"postprocessor.{name}", param

        for name, param in self.criterion.named_parameters():
            yield f"criterion.{name}", param
