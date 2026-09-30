import cv2
import numpy as np
from PIL import Image


class OpenCVPreprocessor:

    def preprocess(self, image: Image.Image) -> Image.Image:

        image_array = np.array(image)

        # RGB → Grayscale
        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        # Noise reduction
        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # Thresholding
        threshold = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        return Image.fromarray(threshold)