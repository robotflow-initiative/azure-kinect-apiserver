import glob

import cv2
import numpy as np

# Read the images
img_paths = glob.glob("./imgs/*.BMP")

for img_path in img_paths:
    img = cv2.imread(img_path)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(img_hsv)

    # get uniques
    unique_colors, counts = np.unique(s, return_counts=True)
    max_s = unique_colors[np.argmax(counts)]

    if max_s == 0:
        continue

    margin = 50
    print(max_s)
    mask = (s > max_s - margin) & (s < max_s + margin)
    mask = np.bitwise_not(mask)
    mask = (mask * 255).astype(np.uint8)
    cv2.imshow("mask_hsv", mask)

    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.medianBlur(mask, 5)
    # cv2.imshow("mask_morphology_1", mask)

    # mask = cv2.erode(mask, np.ones((8, 8), np.uint8))
    # mask = cv2.dilate(mask, np.ones((16, 16), np.uint8))
    # mask = cv2.erode(mask, np.ones((8, 8), np.uint8))
    # cv2.imshow("mask_morphology_2", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_index = np.argmax([cv2.contourArea(c) for c in contours])
    biggest_contour = contours[contour_index]

    result_mask = np.zeros_like(mask)
    cv2.fillPoly(result_mask, [biggest_contour], 255)
    cv2.imshow("mask_contour", result_mask)

    result_mask = cv2.medianBlur(result_mask, 5)
    cv2.imshow("mask_contour_smooth_final", result_mask)

    img[result_mask == 255] = (255, 0, 0)
    cv2.imshow("img", img)

    cv2.waitKey(0)
