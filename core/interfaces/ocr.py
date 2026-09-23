from abc import ABC, abstractmethod
from PIL import Image


class OCRProcessor(ABC):

    @abstractmethod
    def extract_text(self, image: Image.Image) -> str:
        pass