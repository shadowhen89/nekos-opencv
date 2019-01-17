import numpy as np
import cv2

LW_COLOR = [14, 12, 4]
HI_COLOR = [18, 20, 10]

LW_BRIGHT = [10, 10, 230]
HI_BRIGHT = [10, 10, 255]


class GoldDetector:

    def __init__(self):
        pass

    def process(self, img):
        image = img.copy()
        image = cv2.resize(image, None, fx=700/max(image.shape[0]), fy=700/max(image.shape[1]))

        image_blur = cv2.GaussianBlur(image, 7)

        mask_one = cv2.inRange(image, LW_COLOR, HI_COLOR)
        mask_two = cv2.inRange(image, LW_BRIGHT, HI_BRIGHT)
        mask = mask_one + mask_two;

        final = mask.copy()

        return final
