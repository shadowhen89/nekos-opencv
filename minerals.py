import numpy as np, cv2

# These are values for assigning minerals based on position
# According to the rules by FIRST Tech CHallenge game manual for Rover Ruckus,
# there are minerals for sampling; two are silver while one is gold.
# In this case for the prototype, assigning the position by a number would
# tell us which position is gold, silver, or none.
MINERAL_NONE = 0
MINERAL_GOLD = 1
MINERAL_SILVER = 2

# These values are in hsv format
# EX. [hue, saturation, value]
LW_GOLD = [15, 100, 80]
HI_GOLD = [30, 255, 255]

LW_SILVER = [10, 5, 200]
HI_SILVER = [40, 40, 255]

LW_BRIGHT = [170, 255, 255]
HI_BRIGHT = [180, 255, 255]

GOLD_THRESH = 32
SILVER_THRESH = 200


def inital_process(image, size=(700, 700), b_size=(7, 7)):
    working_image = cv2.resize(image, None, fx=size[0] / image.shape[0], fy=size[1] / image.shape[1])
    blurred_image = cv2.GaussianBlur(working_image, b_size, 0)
    hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    return working_image, blurred_image, hsv


def filter_gold(hsv):
    """
    Filters the hsv for gold minerals
    :param hsv:
    :return: Mask
    """
    mask_one = cv2.inRange(hsv, np.array(LW_GOLD), np.array(HI_GOLD))
    mask_two = cv2.inRange(hsv, np.array(LW_BRIGHT), np.array(HI_BRIGHT))
    return mask_one + mask_two


def filter_silver(hsv):
    """
    Filters the hsv for silver minerals
    :param hsv:
    :return: Mask
    """
    mask_one = cv2.inRange(hsv, np.array(LW_SILVER), np.array(HI_SILVER))
    mask_two = cv2.inRange(hsv, np.array(LW_BRIGHT), np.array(HI_BRIGHT))
    return mask_one + mask_two


def clean_mask(mask, shape, size):
    kernel = cv2.getStructuringElement(shape, size)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    return clean


def find_largest_contour(contours):
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    return biggest_contour


def draw_contour(image, contour, color):
    if color is None:
        color = [0, 255, 0]

    cv2.drawContours(image, [contour], -1, 255, 5)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), color)


def process_gold_silver(image):
    """
    Process the image for one gold mineral and one silver mineral
    :param image: Image to process
    :return: image
    """
    working_image, blur, hsv = inital_process(image)

    # GOLD

    # Filter the gold and brightness
    g_mask = filter_gold(hsv)

    # Clean the mask
    g_clean = clean_mask(g_mask, cv2.MORPH_ELLIPSE, (5, 5))

    # Find the contours in the mask
    g_contours, g_hierarchy = cv2.findContours(g_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # SILVER

    # Filter the silver
    s_mask = filter_silver(hsv)

    # Clean the mask
    s_clean = clean_mask(s_mask, cv2.MORPH_ELLIPSE, (5, 5))

    # Find the contours in the mask
    s_contours, s_hierarchy = cv2.findContours(s_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Use largest contours from gold and silver
    # to draw rectangle over the gold and silver mineral
    # One for silver, one for gold
    if len(g_contours) > 0:
        g_biggest_contour = find_largest_contour(g_contours)
        draw_contour(working_image, g_biggest_contour, [0, 255, 0])

    if len(s_contours) > 0:
        s_biggest_contour = find_largest_contour(s_contours)
        draw_contour(working_image, s_biggest_contour, [0, 0, 255])

    return working_image
