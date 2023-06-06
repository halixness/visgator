##
##
##
import requests
import json
from datetime import timedelta
from ratelimit import limits, sleep_and_retry

from typing import Protocol
import sng_parser

from ._graph import Entity, Relation, SceneGraph
from . import _you as you


class _SceneGraphParser(Protocol):
    """A protocol for extractors."""

    def parse(self, sentence: str) -> SceneGraph:
        """Parses a sentence and returns a scene graph."""
        ...


# ------------------------------------------------------------
# GPT based extractor
# ------------------------------------------------------------

SUPPORTED_ENGINES = [
    "PizzaGPT",
    "You"
]

class GPTSceneGraphParser(_SceneGraphParser):
    """A parser that uses GPT."""

    def __init__(self, engine="PizzaGPT") -> None:
        super().__init__()

        if engine not in SUPPORTED_ENGINES: raise Exception("Unsopported parser.")
        else: self.engine = engine


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
    
    @sleep_and_retry
    @limits(calls=10, period=timedelta(seconds=60).total_seconds())
    def _PizzaGPT_parse(self, sentence: str) -> SceneGraph:
        """ Use GPT3.5 with PizzaGPT to extract SceneGraphs """
        try:
            # Quuerying graph
            url = "https://www.pizzagpt.it/api/chat-completion"
            payload = {
                "question": self._create_prompt(sentence),
                "secret": "salame"
            }
            content = requests.post(url, json = payload).json()["answer"]["content"].strip()
            graph = json.loads(content)

            # Constructing SceneGraph obj
            entities = []
            for entity in graph["entities"]:
                entities.append(Entity(entity, entity.split("the")[1].replace(" ", "")))

            relations = []
            for relation in graph["relations"]:
                relations.append(
                    Relation(relation[0], relation[1], relation[2])
                )

            return SceneGraph(entities, relations)            
        except:
            raise Exception("Error in querying PizzaGPT.")
        
        
    @sleep_and_retry
    @limits(calls=10, period=timedelta(seconds=60).total_seconds())
    def _You_parse(self, sentence: str) -> SceneGraph:
        """ Use GPT3.5/4 with YOU to extract SceneGraphs """
        try:
            # Quuerying graph
            response = you.Completion.create(
                prompt = self._create_prompt(sentence),
                detailed = False
            )['response'].split('{')[1].split('}')[0].replace('\\n', '').replace('\\', '').replace(' ', '')
            
            graph = json.loads(f"{{{response}}}")

            # Constructing SceneGraph obj
            entities = []
            for entity in graph["entities"]:
                entities.append(Entity(entity, entity.split("the")[1].replace(" ", "")))

            relations = []
            for relation in graph["relations"]:
                relations.append(
                    Relation(relation[0], relation[1], relation[2])
                )

            return SceneGraph(entities, relations)   
        except:
            raise Exception("Error in querying You GPT.")


    def parse(self, sentence: str) -> SceneGraph:
        parse = getattr(self, f"_{self.engine}_parse")
        return parse(sentence)


# ------------------------------------------------------------
# Spacy based extractor
# ------------------------------------------------------------


class SpacySceneGraphParser(_SceneGraphParser):
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
