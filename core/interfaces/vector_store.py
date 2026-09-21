from abc import ABC, abstractmethod


class VectorStore(ABC):

    @abstractmethod
    def add_documents(self, documents, embeddings):
        pass

    @abstractmethod
    def search(self, query_embedding, k=5):
        pass