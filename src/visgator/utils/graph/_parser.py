##
##
##

from typing import Protocol

import openai
import sng_parser

from ._graph import Relation, SceneGraph


class _SceneGraphParser(Protocol):
    """A protocol for extractors."""

    def parse(self, sentence: str) -> SceneGraph:
        """Parses a sentence and returns a scene graph."""
        ...


# ------------------------------------------------------------
# GPT based extractor
# ------------------------------------------------------------


class GPTSceneGraphParser(_SceneGraphParser):
    """A parser that uses GPT."""

    def __init__(self, api_key: str) -> None:
        openai.api_key = api_key

    def _create_prompt(self, sentence: str) -> str:
        return f"""
            Consider the sentence: {sentence}.
            What are the named entities?
            What are the relations between the named entities?
            Answer only with tuples "[x, action name, y]"  and without passive forms.
            Please be coherent with the name of the actions that occur multiple times.
            Answer by filling a JSON, follow the example:
            Sentence: "the girl is looking ath the table full of drinks"
            Answer:
            {{
            "entities": ["the girl", "the table", "drinks"],
            "relations": [[0, "is on", 1], [1, "full of", 2]]
            }}
        """

    def parse(self, sentence: str) -> SceneGraph:
        raise NotImplementedError()


# ------------------------------------------------------------
# Spacy based extractor
# ------------------------------------------------------------


class SpacySceneGraphParser(_SceneGraphParser):
    """A parser that uses Spacy."""

    def parse(self, sentence: str) -> SceneGraph:
        graph = sng_parser.parse(sentence)

        entities = []
        for entity in graph["entities"]:
            entities.append(entity["span"])

        relations = []
        for relation in graph["relations"]:
            relations.append(
                Relation(relation["subject"], relation["relation"], relation["object"])
            )

        return SceneGraph(entities, relations)
