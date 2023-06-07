##
##
##

from __future__ import annotations

import abc
import enum
import json
from datetime import timedelta

import requests
import sng_parser
from ratelimit import limits, sleep_and_retry

from . import _you as you
from ._graph import Entity, Relation, SceneGraph


class SceneGraphParserType(enum.Enum):
    """An enum that represents the type of scene graph parser."""

    SPACY = "spacy"
    PIZZAGPT = "pizzagpt"
    YOU = "you"


class SceneGraphParser(abc.ABC):
    """A base class for scene graph parsers."""

    @classmethod
    def new(cls, type: SceneGraphParserType) -> SceneGraphParser:
        match type:
            case SceneGraphParserType.SPACY:
                return _SpacySceneGraphParser()
            case SceneGraphParserType.PIZZAGPT:
                return _PizzaGPTSceneGraphParser()
            case SceneGraphParserType.YOU:
                return _YouSceneGraphParser()
            case _:
                raise ValueError(f"Invalid scene graph parser type: {type}.")

    @abc.abstractmethod
    def parse(self, sentence: str) -> SceneGraph:
        """Parses a sentence and returns a scene graph."""


# ------------------------------------------------------------
# GPT based extractor
# ------------------------------------------------------------


def _create_prompt(sentence: str) -> str:
    return f"""
            Consider the sentence: {sentence}.
            What are the named entities?
            What are the relations between the named entities?
            Answer only with tuples "[x, action name, y]"  and without passive forms.
            Please be coherent with the name of the actions that occur multiple times.
            Answer by filling a JSON, follow the example:
            Sentence: "the girl is looking at the table full of drinks"
            Answer:
            {{
            "entities": ["the girl", "the table", "drinks"],
            "relations": [[0, "is on", 1], [1, "full of", 2]]
            }}
        """


class _PizzaGPTSceneGraphParser(SceneGraphParser):
    """A parser that uses PizzaGPT."""

    @sleep_and_retry  # type: ignore
    @limits(calls=10, period=timedelta(seconds=60).total_seconds())  # type: ignore
    def parse(self, sentence: str) -> SceneGraph:
        try:
            # Quuerying graph
            url = "https://www.pizzagpt.it/api/chat-completion"
            payload = {"question": _create_prompt(sentence), "secret": "salame"}
            content = (
                requests.post(url, json=payload).json()["answer"]["content"].strip()
            )
            graph = json.loads(content)

            # Constructing SceneGraph obj
            entities = []
            for entity in graph["entities"]:
                entities.append(Entity(entity, entity.split("the")[1].replace(" ", "")))

            relations = []
            for relation in graph["relations"]:
                relations.append(Relation(relation[0], relation[1], relation[2]))

            return SceneGraph(entities, relations)
        except Exception:
            raise RuntimeError("Error in querying PizzaGPT.")


class _YouSceneGraphParser(SceneGraphParser):
    """A parser that uses You GPT."""

    @sleep_and_retry  # type: ignore
    @limits(calls=10, period=timedelta(seconds=60).total_seconds())  # type: ignore
    def parse(self, sentence: str) -> SceneGraph:
        try:
            # Quuerying graph
            response = (
                you.Completion.create(prompt=_create_prompt(sentence))
                .split("{")[1]
                .split("}")[0]
                .replace("\\n", "")
                .replace("\\", "")
                .replace(" ", "")
            )

            graph = json.loads(f"{{{response}}}")

            # Constructing SceneGraph obj
            entities = []
            for entity in graph["entities"]:
                entities.append(Entity(entity, entity.split("the")[1].replace(" ", "")))

            relations = []
            for relation in graph["relations"]:
                relations.append(Relation(relation[0], relation[1], relation[2]))

            return SceneGraph(entities, relations)
        except Exception:
            raise RuntimeError("Error in querying You GPT.")


# ------------------------------------------------------------
# Spacy based extractor
# ------------------------------------------------------------


class _SpacySceneGraphParser(SceneGraphParser):
    """A parser that uses Spacy."""

    def parse(self, sentence: str) -> SceneGraph:
        graph = sng_parser.parse(sentence)

        entities = []
        for entity in graph["entities"]:
            entities.append(Entity(entity["span"], entity["head"]))

        relations = []
        for relation in graph["relations"]:
            relations.append(
                Relation(relation["subject"], relation["relation"], relation["object"])
            )

        return SceneGraph(entities, relations)
