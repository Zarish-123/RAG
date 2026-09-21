from pydantic import BaseModel
from typing import List, Dict, Any


class QuestionRequest(BaseModel):

    question: str
    k: int = 10


class Source(BaseModel):

    content: str
    metadata: Dict[str, Any] = {}


class AnswerResponse(BaseModel):

    question: str
    answer: str
    sources: List[Source] = []