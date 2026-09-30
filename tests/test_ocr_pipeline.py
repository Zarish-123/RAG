from ingestion.test_images_extract import PDFImageExtractor
from ingestion.test_opencv_preprocessor import OpenCVPreprocessor
from ingestion.test_ocr_processor import TesseractOCRProcessor


pdf_path = "data/document.pdf"


extractor = PDFImageExtractor()
preprocessor = OpenCVPreprocessor()
ocr = TesseractOCRProcessor()


images = extractor.extract(pdf_path)

print("Total pages:", len(images))


for page_number, image in enumerate(images, start=1):

    print(f"\nProcessing page {page_number}...")

    processed_image = preprocessor.preprocess(image)

    text = ocr.extract_text(processed_image)

    print("Extracted Text:")
    print(text)