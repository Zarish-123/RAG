from typing import Protocol

from models import ApiResult


class ApiClientInterface(Protocol):

    def ask(
        self,
        question: str,
        k: int
    ) -> ApiResult:
        ...