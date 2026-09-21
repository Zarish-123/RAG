from dataclasses import dataclass


@dataclass
class AppConfig:
    api_url: str = "http://127.0.0.1:8000/ask"
    page_title: str = "RAG Assistant"
    page_icon: str = "🤖"
    document_name: str = "machine learning.pdf"
    default_top_k: int = 10
    max_top_k: int = 20