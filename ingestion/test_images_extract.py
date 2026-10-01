import io

import pymupdf
from PIL import Image

from core.interfaces.test_image_extractor import ImageExtractor


class PDFImageExtractor(ImageExtractor):

    def extract(self, source: str) -> list[Image.Image]:

        pdf = pymupdf.open(source)

        images = []

        for page in pdf:

            matrix = pymupdf.Matrix(2, 2)

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            image_bytes = pixmap.tobytes("png")

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            images.append(image)

        pdf.close()

        return images