##
##
##

import ast
import enum
import os
from typing import Any

import openai

from deepsight.data.structs import SceneGraph


class GPTModel(enum.Enum):
    GPT3_5 = "gpt-3.5-turbo"
    GPT4 = "gpt-4"

    def openai_model(self) -> str:
        return self.value


class SceneGraphParser:
    def __init__(
        self,
        api_key: str | None = None,
        model: GPTModel = GPTModel.GPT3_5,
        temperature: float = 0.2,
    ) -> None:
        """Initializes the parser with the given parameters.

        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. If not provided, the token will be read from the
            environment variable OPENAI_API_KEY.
        model : GPTModel, optional
            The GPT model to use. Defaults to GPTModel.GPT3_5.
        temperature : float, optional
            The temperature to use when sampling from the model. Should be between
            0 and 2, where higher values will make the output more random, while
            lower values will make the output more focused and deterministic.
            Defaults to 0.2.
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "No OpenAI API key provided. Please provide a key or set the "
                    "environment variable OPENAI_API_KEY."
                )

        openai.api_key = api_key
        self.model = model
        self.temperature = temperature

    def _build_requests(self, captions: list[str]) -> str:
        output = ""
        for caption in captions:
            output += f"<caption>{caption}</caption>\n"

        return output

    def _build_examples(self, graphs: list[SceneGraph]) -> str:
        output = ""
        for graph in graphs:
            output += f"<json>{graph.to_dict()}</json>\n"

        return output

    def _match_entity(self, entity: str, entities: list[dict[str, Any]]) -> int | None:
        """Matches the given entity to an entity in the list of entities."""

        for idx, ent in enumerate(entities):
            if entity in ent["phrase"]:
                return idx

        return None

    def _get_entity_index(
        self, entity: str | int | None, entities: list[dict[str, Any]]
    ) -> int | None:
        if entity is None:
            return None

        entity_idx: int
        if isinstance(entity, int):
            entity_idx = entity
        elif entity.isdigit():
            entity_idx = int(entity)
        else:
            idx = self._match_entity(entity, entities)
            if idx is None:
                entities.append({"noun": entity, "phrase": entity})
                entity_idx = len(entities) - 1
            else:
                entity_idx = idx

        if entity_idx >= len(entities):
            return None

        return entity_idx

    def _postprocess(self, output: dict[str, Any]) -> SceneGraph | None:
        entities = output.get("entities", [])
        if len(entities) == 0:
            return None

        triplets = output.get("triplets", [])
        new_triplets = []
        for triplet in triplets:
            subj = self._get_entity_index(triplet.get("subject"), entities)
            obj = self._get_entity_index(triplet.get("object"), entities)

            match (subj, obj):
                case (None, None):
                    continue
                case (None, obj):
                    entities[obj]["phrase"] = (
                        entities[obj]["phrase"] + " " + triplet["relation"]
                    )
                case (subj, None):
                    entities[subj]["phrase"] = (
                        entities[subj]["phrase"] + " " + triplet["relation"]
                    )
                case (subj, obj):
                    new_triplets.append(
                        {
                            "subject": subj,
                            "object": obj,
                            "relation": triplet["relation"],
                        }
                    )

        return SceneGraph.from_dict({"entities": entities, "triplets": new_triplets})

    async def parse(
        self, examples: list[tuple[str, SceneGraph]], captions: list[str]
    ) -> list[tuple[str, SceneGraph | None]]:
        """Parses the given captions into scene graphs.

        Parameters
        ----------
        examples : list[tuple[str, SceneGraph]]
            A list of examples to use for the prompt. Each example is a tuple
            consisting of a caption and the corresponding scene graph.
        captions : list[str]
            A list of captions to parse into scene graphs.

        Returns
        -------
        list[tuple[str, SceneGraph | None]]
            A list of tuples consisting of the original caption and the parsed
            scene graph. If the parsing fails due to formatting issues, the scene
            graph will be `None`.

        Raises
        ------
        RuntimeError
            If the parsing fails.
        openai.error.OpenAIError
            The error returned by the OpenAI API.
        """

        try:
            res = await openai.ChatCompletion.acreate(
                model=self.model.openai_model(),
                temperature=self.temperature,
                n=1,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": self._build_requests([cap for cap, _ in examples]),
                    },
                    {
                        "role": "assistant",
                        "content": self._build_examples(
                            [graph for _, graph in examples]
                        ),
                    },
                    {
                        "role": "user",
                        "content": self._build_requests(captions),
                    },
                ],
            )
        except openai.error.OpenAIError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Input: {captions}") from e

        response: str = res["choices"][0]["message"]["content"]
        outputs = response.split("\n")

        results: list[tuple[str, SceneGraph | None]] = []
        for caption, output in zip(captions, outputs):
            start = output.find("{")
            end = output.rfind("}")
            output = output[start : end + 1]

            try:
                output_dict = ast.literal_eval(output)
                graph = self._postprocess(output_dict)
                if graph is not None:
                    results.append((caption, graph))
                else:
                    results.append((caption, None))
            except Exception as e:
                results.append((caption, None))
                print(f"Input: {caption} | Output: {output}")
                print(f"Exception: {e}")

        return results


system = """\
You will be provided with a set of captions each describing a region in an image. \
For each region, first identify the entities, like people, objects or places, present in the region Specify both a single noun and a phrase that describes the entity. \
Then, identify the triplets of subject, relation and object that describe the relationships between the entities. \
"""  # noqa: E501
