import pytesseract
from PIL import Image

from core.interfaces.ocr import OCRProcessor


class TesseractOCRProcessor(OCRProcessor):

    def __init__(self, language: str = "eng"):

        self.language = language

        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

    def extract_text(self, image: Image.Image) -> str:

        return pytesseract.image_to_string(
            image,
            lang=self.language
        )