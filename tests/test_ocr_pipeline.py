from ingestion.test_images_extract import PDFImageExtractor
from ingestion.test_opencv_preprocessor import OpenCVPreprocessor
from ingestion.test_ocr_processor import TesseractOCRProcessor


# --------------------------------
# PDF path
# --------------------------------

pdf_path = "data/document.pdf"


# --------------------------------
# Create objects
# --------------------------------

extractor = PDFImageExtractor()

preprocessor = OpenCVPreprocessor()

ocr = TesseractOCRProcessor()


# --------------------------------
# PDF → Images
# --------------------------------

images = extractor.extract(
    pdf_path
)

print(
    "Total pages:",
    len(images)
)


# --------------------------------
# Process every page
# --------------------------------

for page_number, image in enumerate(
    images,
    start=1
):

    print(
        f"\nProcessing page {page_number}..."
    )


    # --------------------------------
    # OpenCV preprocessing
    # --------------------------------

    processed_image = preprocessor.preprocess(
        image
    )


    # --------------------------------
    # OCR
    # --------------------------------

    text = ocr.extract_text(
        processed_image
    )


    print("\nExtracted Text:")
    print("-" * 60)
    print(text)
    print("-" * 60)