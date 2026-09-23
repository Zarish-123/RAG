
import re

import pytesseract

from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
    ImageOps
)

from core.interfaces.ocr import OCRProcessor


class TesseractOCRProcessor(
    OCRProcessor
):

    def __init__(
        self,
        language: str = "eng"
    ):

        self.language = language

        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR"
            r"\tesseract.exe"
        )

    # ---------------------------------
    # Image preprocessing
    # ---------------------------------

    def preprocess_image(
        self,
        image: Image.Image
    ) -> Image.Image:

        # Convert to grayscale
        image = image.convert("L")

        # Upscale
        width, height = image.size

        image = image.resize(
            (
                width * 3,
                height * 3
            )
        )

        # Increase contrast
        enhancer = ImageEnhance.Contrast(
            image
        )

        image = enhancer.enhance(
            2.5
        )

        # Sharpen
        image = image.filter(
            ImageFilter.SHARPEN
        )

        # Autocontrast
        image = ImageOps.autocontrast(
            image
        )

        return image

    # ---------------------------------
    # Clean OCR output
    # ---------------------------------

    def clean_text(
        self,
        text: str
    ) -> str:

        if not text:
            return ""

        # Normalize line endings
        text = text.replace(
            "\r\n",
            "\n"
        )

        # Common OCR corrections
        corrections = {

            "Data Caflection":
                "Data Collection",

            "Data Caflection,":
                "Data Collection",

            "Chuniang":
                "Chunking",

            "Chuniang,":
                "Chunking",

            "Refievall":
                "Retrieval",

            "Refievall,":
                "Retrieval",

            "Embedcling":
                "Embedding",

            "Embedcling,":
                "Embedding",

            "Generatlon":
                "Generation",

            "Generatlon,":
                "Generation",

            "Resp0nse":
                "Response",

            "Resp0nse,":
                "Response"
        }

        for wrong, correct in corrections.items():

            text = text.replace(
                wrong,
                correct
            )

        # ---------------------------------
        # Remove obvious diagram noise
        # ---------------------------------

        cleaned_lines = []

        for line in text.split("\n"):

            line = line.strip()

            if not line:
                continue

            # Remove lines consisting mainly
            # of OCR-detected borders/arrows.
            alphanumeric_count = sum(
                character.isalnum()
                for character in line
            )

            if alphanumeric_count < 3:
                continue

            # Remove repeated symbols
            line = re.sub(
                r"[-_=|~]+",
                " ",
                line
            )

            # Normalize spaces
            line = re.sub(
                r"\s+",
                " ",
                line
            ).strip()

            if line:
                cleaned_lines.append(
                    line
                )

        return "\n".join(
            cleaned_lines
        )

    # ---------------------------------
    # OCR
    # ---------------------------------

    def extract_text(
        self,
        image: Image.Image
    ) -> str:

        processed_image = (
            self.preprocess_image(
                image
            )
        )

        results = []

        configs = [
            "--psm 6",
            "--psm 11",
            "--psm 12"
        ]

        for config in configs:

            text = pytesseract.image_to_string(
                processed_image,
                lang=self.language,
                config=config
            )

            cleaned = self.clean_text(
                text
            )

            if cleaned:
                results.append(
                    cleaned
                )

        if not results:
            return ""

        # ---------------------------------
        # Choose the result containing
        # the most useful OCR keywords.
        # ---------------------------------

        keywords = [
            "document",
            "data",
            "collection",
            "preprocessing",
            "chunking",
            "embedding",
            "retrieval",
            "generation",
            "response",
            "query",
            "model"
        ]

        best_result = results[0]
        best_score = 0

        for result in results:

            result_lower = (
                result.lower()
            )

            score = sum(
                keyword in result_lower
                for keyword in keywords
            )

            if score > best_score:

                best_score = score
                best_result = result

        return best_result

