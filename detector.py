import numpy as np
import cv2

LW_COLOR = [17, 100, 80]
HI_COLOR = [30, 255, 255]

LW_BRIGHT = [170, 255, 255]
HI_BRIGHT = [180, 255, 255]

THRESH = 32


class GoldDetector:

    def __init__(self):
        pass

    def process(self, img):
        image = cv2.resize(img, None, fx=700/img.shape[0], fy=700/img.shape[1])

        blur = cv2.GaussianBlur(image, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        mask_one = cv2.inRange(hsv, np.array(LW_COLOR), np.array(HI_COLOR))
        mask_two = cv2.inRange(hsv, np.array(LW_BRIGHT), np.array(HI_BRIGHT))

        mask = mask_one + mask_two
        ret, mask = cv2.threshold(mask, THRESH, 128, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        open = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        contours, hierarchy = cv2.findContours(open, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

        if (len(contour_sizes) > 0):
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            cv2.drawContours(image, [biggest_contour], -1, 255, 5)
            x, y, w, h = cv2.boundingRect(biggest_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), [0, 255, 0])

        return image
