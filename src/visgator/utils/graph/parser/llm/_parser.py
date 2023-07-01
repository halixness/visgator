##
##
##

import json
from typing import Generator, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing_extensions import Self

from visgator.utils.graph import Entity, Relation, SceneGraph
from visgator.utils.graph.parser import Parser as _Parser

from ._config import Config


class Parser(_Parser):
    def __init__(self, config: Config) -> None:
        self._name = config.model

        with open(config.prompt, "r") as f:
            self._prompt: str = f.read()

        tokenizer = AutoTokenizer.from_pretrained(config.model)
        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            batch_size=config.batch_size,
        )

        self._pipe.tokenizer.pad_token_id = model.config.eos_token_id

    @classmethod
    def new(cls, config: Config) -> Self:  # type: ignore
        return cls(config)

    @property
    def name(self) -> str:
        return self._name

    def _create_prompt(self, sentence: str) -> str:
        sentence1 = "the girl looking at the table full of drinks"
        example1 = {
            "entities": [
                ("the girl", "girl"),
                ("the table", "table"),
                ("drinks", "drinks"),
            ],
            "relations": [
                (0, "looking at", 1),
                (1, "full of", 2),
            ],
        }

        sentence2 = (
            "the man wearing a long sleeved white shirt and a pair of blue jeans "
            "catching a freesbie"
        )
        example2 = {
            "entities": [
                ("the man", "man"),
                ("a long sleeved white shirt", "shirt"),
                ("a pair of blue jeans", "jeans"),
                ("a freesbie", "freesbie"),
            ],
            "relations": [
                (0, "wearing", 1),
                (0, "wearing", 2),
                (0, "catching", 3),
            ],
        }

        sentence3 = "Skateboarder in green"
        example3 = {
            "entities": [
                ("Skateboarder", "Skateboarder"),
                ("green clothes", "clothes"),
            ],
            "relations": [
                (0, "in", 1),
            ],
        }

        sentence4 = "glass far right"
        example4 = {
            "entities": [("glass far right", "glass")],
            "relations": [],
        }

        sentence5 = "2nd to theleft brown horse drinking"
        example5 = {
            "entities": [
                ("brown horse drinking", "horse"),
                ("leftmost brown horse", "horse"),
            ],
            "relations": [
                (0, "at the right of", 1),
            ],
        }

        return self._prompt.format(
            sentence1=sentence1,
            example1=json.dumps(example1),
            sentence2=sentence2,
            example2=json.dumps(example2),
            sentence3=sentence3,
            example3=json.dumps(example3),
            sentence4=sentence4,
            example4=json.dumps(example4),
            sentence5=sentence5,
            example5=json.dumps(example5),
            sentence6=sentence,
        )

    def parse(self, sentences: Iterable[str]) -> Generator[SceneGraph, None, None]:
        prompts = (self._create_prompt(s) for s in sentences)

        generator = self._pipe(
            prompts,
            max_length=1000,
            num_return_sequences=1,
            do_sample=False,
            # top_k=10,
            return_full_text=False,
        )

        for output in generator:
            generated = output[0]["generated_text"]
            generated = generated.split("\n")[0]
            gen_json = json.loads(generated)

            entities = []
            for span, head in gen_json["entities"]:
                entities.append(Entity(span=span, head=head))

            relations = []
            for s, pred, o in gen_json["relations"]:
                relations.append(Relation(s, pred, o))

            yield SceneGraph(entities=entities, relations=relations)
