from abc import ABC, abstractmethod


class Retriever(ABC):

    @abstractmethod
    def retrieve(self, query, k=5):
        pass