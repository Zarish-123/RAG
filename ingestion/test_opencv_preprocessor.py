import cv2
import numpy as np
from PIL import Image


class OpenCVPreprocessor:

    def preprocess(self, image: Image.Image) -> Image.Image:

        # PIL → NumPy
        image_array = np.array(image)

        # RGB → Grayscale
        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        # Light noise removal
        blur = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        # Otsu thresholding
        threshold = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        # NumPy → PIL
        processed_image = Image.fromarray(
            threshold
        )

        return processed_image