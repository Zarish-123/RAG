from abc import ABC, abstractmethod
from PIL import Image


class ImageExtractor(ABC):

    @abstractmethod
    def extract(self, source: str) -> list[Image.Image]:
        pass