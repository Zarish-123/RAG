from pypdf import PdfReader

from core.interfaces.document_loader import DocumentLoader
from core.interfaces.models.document import Document
from core.interfaces.ocr import OCRProcessor

from ingestion.image_extractor import PDFImageExtractor


class PDFLoader(
    DocumentLoader
):

    def __init__(
        self,
        ocr_processor: OCRProcessor | None = None,
        image_extractor: PDFImageExtractor | None = None
    ):

        self.ocr_processor = (
            ocr_processor
        )

        self.image_extractor = (
            image_extractor
        )

    def load(
        self,
        file_path
    ):

        reader = PdfReader(
            file_path
        )

        documents = []

        for page_number, page in enumerate(
            reader.pages
        ):

            # ---------------------------------
            # 1. Normal PDF text
            # ---------------------------------

            text = (
                page.extract_text()
                or ""
            )

            ocr_texts = []

            # ---------------------------------
            # 2. Extract embedded images
            # ---------------------------------

            if (
                self.ocr_processor
                and self.image_extractor
            ):

                images = (
                    self.image_extractor
                    .extract_embedded_images(
                        file_path,
                        page_number
                    )
                )

                print(
                    f"Page {page_number + 1}: "
                    f"{len(images)} actual image(s) found"
                )

                # ---------------------------------
                # 3. OCR images
                # ---------------------------------

                for image_number, image in enumerate(
                    images,
                    start=1
                ):

                    print(
                        f"   OCR processing "
                        f"image {image_number}: "
                        f"{image.size}"
                    )

                    ocr_text = (
                        self.ocr_processor
                        .extract_text(
                            image
                        )
                    )

                    if ocr_text:

                        print(
                            "\n========== OCR TEXT =========="
                        )

                        print(
                            ocr_text
                        )

                        print(
                            "==============================\n"
                        )

                        ocr_texts.append(
                            ocr_text
                        )

                    else:

                        print(
                            "   OCR returned "
                            "no text."
                        )

            # ---------------------------------
            # 4. Combine text + OCR
            # ---------------------------------

            if ocr_texts:

                combined_ocr = (
                    "\n\n".join(
                        ocr_texts
                    )
                )

                text = (
                    f"{text}\n\n"
                    f"{combined_ocr}"
                ).strip()

            # ---------------------------------
            # 5. Fallback for scanned page
            # ---------------------------------

            elif (
                not text.strip()
                and self.ocr_processor
                and self.image_extractor
            ):

                print(
                    f"Page {page_number + 1}: "
                    "No text or embedded image found."
                )

                print(
                    "Using full-page OCR..."
                )

                page_image = (
                    self.image_extractor
                    .render_page(
                        file_path,
                        page_number
                    )
                )

                page_ocr = (
                    self.ocr_processor
                    .extract_text(
                        page_image
                    )
                )

                if page_ocr:

                    print(
                        "\n========== PAGE OCR =========="
                    )

                    print(
                        page_ocr
                    )

                    print(
                        "==============================\n"
                    )

                    text = page_ocr

            # ---------------------------------
            # 6. Create Document
            # ---------------------------------

            if text.strip():

                documents.append(
                    Document(
                        content=text,
                        metadata={
                            "page": page_number + 1,
                            "source": file_path
                        }
                    )
                )

        return documents