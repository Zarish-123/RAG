from PIL import Image, ImageDraw

from ingestion.ocr_processor import TesseractOCRProcessor


# Create a simple test image
image = Image.new(
    "RGB",
    (800, 200),
    "white"
)

draw = ImageDraw.Draw(image)

draw.text(
    (50, 70),
    "Machine Learning is a subset of Artificial Intelligence.",
    fill="black"
)


# Create OCR processor
ocr = TesseractOCRProcessor()


# Extract text
text = ocr.extract_text(image)


print("\n==============================")
print("OCR RESULT")
print("==============================")
print(text)