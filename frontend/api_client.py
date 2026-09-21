import requests

from interfaces import ApiClientInterface
from models import ApiResult


class FastApiClient(ApiClientInterface):

    def __init__(self, api_url: str):
        self.api_url = api_url

    def ask(
        self,
        question: str,
        k: int
    ) -> ApiResult:

        try:
            response = requests.post(
                self.api_url,
                json={
                    "question": question,
                    "k": k
                },
                timeout=120
            )

            response.raise_for_status()

            data = response.json()

            return ApiResult(
                answer=data.get(
                    "answer",
                    "No answer returned."
                ),
                sources=data.get(
                    "sources",
                    []
                )
            )

        except requests.exceptions.RequestException as error:

            return ApiResult(
                answer="",
                sources=[],
                success=False,
                error=str(error)
            )