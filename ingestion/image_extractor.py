import io
import os

import pymupdf
from PIL import Image


class PDFImageExtractor:

    def __init__(
        self,
        dpi: int = 200
    ):
        self.dpi = dpi

    def extract_embedded_images(
        self,
        file_path: str,
        page_number: int
    ) -> list[Image.Image]:

        pdf = pymupdf.open(file_path)

        images = []

        try:

            page = pdf.load_page(
                page_number
            )

            image_list = page.get_images(
                full=True
            )

            processed_xrefs = set()

            for image_info in image_list:

                xref = image_info[0]

                if xref in processed_xrefs:
                    continue

                processed_xrefs.add(xref)

                try:

                    image_data = pdf.extract_image(
                        xref
                    )

                    image_bytes = image_data["image"]

                    image = Image.open(
                        io.BytesIO(image_bytes)
                    ).convert("RGB")

                    width, height = image.size

                    # ---------------------------------
                    # Detect full-page image
                    # ---------------------------------

                    page_width = page.rect.width
                    page_height = page.rect.height

                    image_ratio = width / height
                    page_ratio = (
                        page_width / page_height
                    )

                    ratio_difference = abs(
                        image_ratio - page_ratio
                    )

                    if ratio_difference < 0.05:

                        print(
                            f"   Ignoring full-page image: "
                            f"{image.size}"
                        )

                        continue

                    # ---------------------------------
                    # Save actual embedded image
                    # ---------------------------------

                    os.makedirs(
                        "data/extracted_images",
                        exist_ok=True
                    )

                    image_path = (
                        f"data/extracted_images/"
                        f"page_{page_number + 1}"
                        f"_image_{len(images) + 1}.png"
                    )

                    image.save(
                        image_path
                    )

                    print(
                        f"   Keeping image: "
                        f"{image.size}"
                    )

                    print(
                        f"   Saved image: "
                        f"{image_path}"
                    )

                    images.append(
                        image
                    )

                except Exception as e:

                    print(
                        f"Could not extract image "
                        f"{xref}: {e}"
                    )

            return images

        finally:

            pdf.close()

    def render_page(
        self,
        file_path: str,
        page_number: int
    ) -> Image.Image:

        pdf = pymupdf.open(
            file_path
        )

        try:

            page = pdf.load_page(
                page_number
            )

            zoom = self.dpi / 72

            matrix = pymupdf.Matrix(
                zoom,
                zoom
            )

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            image_bytes = pixmap.tobytes(
                "png"
            )

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            return image

        finally:

            pdf.close()