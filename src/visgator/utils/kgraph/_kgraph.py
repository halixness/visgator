import json
import requests
from abc import ABC, abstractmethod
from . import _you as you

class KnowledgeGraphExtractor:

    @abstractmethod
    def parse_sentence(self, sentence):
        pass

    def format_prompt(self, sentence):
        return """
            Consider the sentence: """ + sentence + """.
            What are the named entities?
            What are the relationships between the named entities? 
            Answer only with tuples "[x, action name, y]"  and without passive forms. 
            Please be coherent with the name of the actions that occur multiple times.  
            Answer by filling a JSON, follow the example:
            Sentence: "The apple is on the table"
            Answer: 
            {
            "entities": ["apple", "table"],
            "relationships": ["apple", "is on", "table"],
            "unique_relationships_types": ["is on"]
            }
        """

class PizzaGPTKGE(KnowledgeGraphExtractor):

    def parse_sentence(self, sentence) -> object:
        """ Using PizzaGPT open post APIs"""
        
        try:
            url = "https://www.pizzagpt.it/api/chat-completion"
            payload = {
                "question": self.format_prompt(sentence),
                "secret": "salame"
            }
            return json.loads(
                requests.post(url, json = payload).text.strip()
            )["answer"]["content"]
        except:
            raise Exception("Error in querying PizzaGPT.")
        
class YouKGE(KnowledgeGraphExtractor):

    def parse_sentence(self, sentence) -> object:
        """ Using YOU Api"""
        
        try:
            response = you.Completion.create(
                prompt = self.format_prompt(sentence),
                detailed = False
            )['response'].split('{')[1].split('}')[0].replace('\\n', '').replace('\\', '').replace(' ', '')
            
            return json.loads(f"{{{response}}}")
        except:
            raise Exception("Error in querying You GPT.")