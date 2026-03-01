import cv2
import numpy as np

from app.operators.base import BaseOperator


class SobelDerivative(BaseOperator):
    def compute(self, image: np.ndarray) -> np.ndarray:
        direction = self.params.get("type", "HORIZONTAL")
        ddepth = int(self.params.get("ddepth", 0))
        if ddepth == 0:
            ddepth = cv2.CV_64F

        if direction == "HORIZONTAL":
            result = cv2.Sobel(image, ddepth, 1, 0)
        elif direction == "VERTICAL":
            result = cv2.Sobel(image, ddepth, 0, 1)
        else:
            sobel_x = cv2.Sobel(image, ddepth, 1, 0)
            sobel_y = cv2.Sobel(image, ddepth, 0, 1)
            result = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

        return np.uint8(np.absolute(result))
