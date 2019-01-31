import numpy as np, cv2

# These values are in hsv format
# EX. [hue, saturation, value]
LW_GOLD = [17, 100, 80]
HI_GOLD = [30, 255, 255]

LW_SILVER = [10, 5, 200]
HI_SILVER = [40, 40, 255]

LW_BRIGHT = [170, 255, 255]
HI_BRIGHT = [180, 255, 255]

GOLD_THRESH = 32
SILVER_THRESH = 200


def process_gold(image):
    working_image = cv2.resize(image, None, fx=700 / image.shape[0], fy=700 / image.shape[1])

    blur = cv2.GaussianBlur(working_image, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask_one = cv2.inRange(hsv, np.array(LW_GOLD), np.array(HI_GOLD))
    mask_two = cv2.inRange(hsv, np.array(LW_BRIGHT), np.array(HI_BRIGHT))

    mask = mask_one + mask_two
    ret, mask = cv2.threshold(mask, GOLD_THRESH, 128, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

    if (len(contour_sizes) > 0):
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        cv2.drawContours(working_image, [biggest_contour], -1, 255, 5)
        x, y, w, h = cv2.boundingRect(biggest_contour)
        cv2.rectangle(working_image, (x, y), (x + w, y + h), [0, 255, 0])

    return working_image


def process_silver(image):
    working_image = cv2.resize(image, None, fx=700 / image.shape[0], fy=700 / image.shape[1])
    blurred_image = cv2.GaussianBlur(working_image, (7, 7), 0)
    hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # Filter out the color
    mask_one = cv2.inRange(hsv, np.array(LW_SILVER), np.array(HI_SILVER))
    mask_two = cv2.inRange(hsv, np.array(LW_BRIGHT), np.array(HI_BRIGHT))
    mask = mask_one + mask_two

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

    if (len(contour_sizes) > 0):
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        cv2.drawContours(working_image, [biggest_contour], -1, 255, 5)
        x, y, w, h = cv2.boundingRect(biggest_contour)
        cv2.rectangle(working_image, (x, y), (x + w, y + h), [0, 255, 0])

    return working_image
