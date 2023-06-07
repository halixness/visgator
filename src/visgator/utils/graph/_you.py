##
##
##

from uuid import uuid4

from tls_client import Session


class Completion:
    @staticmethod
    def create(prompt: str) -> str:
        client = Session(client_identifier="chrome_108")
        client.headers = {
            "authority": "you.com",
            "accept": "text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "referer": "https://you.com/search?q=who+are+you&tbm=youchat",
            "sec-ch-ua": '"Not_A Brand";v="99", "Google Chrome";v="109 ",'
            '"Chromium";v="109"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "cookie": f"safesearch_guest=Moderate; uuid_guest={str(uuid4())}",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        }

        response = client.get(
            "https://you.com/api/streamingSearch",
            params={
                "q": prompt,
                "page": 1,
                "count": 10,
                "safeSearch": "Moderate",
                "onShoppingPage": False,
                "mkt": "",
                "responseFilter": "WebPages,Translations,TimeZone,Computation,"
                "RelatedSearches",
                "domain": "youchat",
                "queryTraceId": str(uuid4()),
                "chat": "[]",  # {"question":"","answer":" '"}
            },
        )

        text = response.text.split(
            '}]}\n\nevent: youChatToken\ndata: {"youChatToken": "'
        )[-1]
        text = text.replace('"}\n\nevent: youChatToken\ndata: {"youChatToken": "', "")
        text = text.replace("event: done\ndata: I'm Mr. Meeseeks. Look at me.\n\n", "")
        text = text[:-4]  # trims '"}', along with the last two remaining newlines

        return text  # type: ignore
