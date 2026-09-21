from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ApiResult:
    answer: str
    sources: List[Dict[str, Any]]
    success: bool = True
    error: Optional[str] = None