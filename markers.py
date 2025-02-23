from __future__ import division, print_function, unicode_literals
from cv2 import cvtColor, GaussianBlur, threshold, findContours, contourArea, boundingRect, arcLength
from cv2 import COLOR_BGR2GRAY, THRESH_BINARY_INV, THRESH_OTSU, RETR_TREE, CHAIN_APPROX_NONE
from cv2 import boundingRect, fitEllipse, warpAffine, getAffineTransform
from cv2 import circle, LINE_AA
from cv2 import resize, INTER_CUBIC, putText, FONT_HERSHEY_PLAIN
from ellipse_helpers import approx_ellipse_perimeter, find_nearest_point_on_ellipse, rotate_point
import numpy as np

def contour_sanity_check(contour, image_height, point_d=0.00635):
    # point_d set to 0.00635 meters (1/4 inch)
    x, y, w, h = boundingRect(contour)
    lb = 1.3 / image_height
    ub = 2 * 2.2 / image_height
    if max(w, h) * ub < point_d or max(w, h) * lb > point_d:
        return False
    if contourArea(contour) > np.pi * (point_d / 2 / lb) ** 2:
        return False
    if arcLength(contour, True) * lb > np.pi * point_d:
        return False
    if arcLength(contour, True) * ub < 2 * point_d:
        return False
    epsilon_factor = 1.5
    if arcLength(contour, True) > epsilon_factor * approx_ellipse_perimeter(w, h):
        return False
    if len(contour) < 5:
        return False
    ellipse = fitEllipse(contour)
    if ellipse[1][0] <= 0 or ellipse[1][1] <= 0:
        return False
    if min(ellipse[1]) < 0.1 * max(ellipse[1]):
        return False
    quad_dist = 0
    for p in contour:
        tp = p[0][0] - ellipse[0][0], p[0][1] - ellipse[0][1]
        rtp = rotate_point(tp, (0, 0), -ellipse[2] + 90)
        poe = find_nearest_point_on_ellipse(ellipse[1][1] / 2, ellipse[1][0] / 2, rtp)
        poer = rotate_point(poe, (0, 0), ellipse[2] - 90)
        poert = poer[0] + ellipse[0][0], poer[1] + ellipse[0][1]
        quad_dist += (p[0][0] - poert[0]) ** 2 + (p[0][1] - poert[1]) ** 2
    if quad_dist / len(contour) > 1.0:
        return False
    return True

def find_marker_ellipses(im):
    im_gray = cvtColor(im, COLOR_BGR2GRAY)
    im_blur = GaussianBlur(im_gray, (3, 3), 0)
    ret, th = threshold(im_blur, 0, 255, THRESH_BINARY_INV + THRESH_OTSU)
    contours, hierarchy = findContours(th, RETR_TREE, CHAIN_APPROX_NONE)
    points = []
    origins = []
    ellipses = []
    print("Found {} contours".format(len(contours)))
    for cnt in contours:
        if contour_sanity_check(cnt, im.shape[0], point_d=0.00635):
            x, y, w, h = boundingRect(cnt)
            ellipse = fitEllipse(cnt)
            points.append(im_gray[y:y + h, x:x + w])
            origins.append((x, y))
            ellipses.append(ellipse)

    print("Found {} points".format(len(points)))
    return points, origins, ellipses

def unskew_point(imc, origin, ellipse):
    center_in_cut = ellipse[0][0] - origin[0], ellipse[0][1] - origin[1]
    source_points = np.float32([center_in_cut,
                                [center_in_cut[0] + np.sin(ellipse[2] / 180 * np.pi) * ellipse[1][1] / 2,
                                 center_in_cut[1] - np.cos(ellipse[2] / 180 * np.pi) * ellipse[1][1] / 2],
                                [center_in_cut[0] - np.cos(ellipse[2] / 180 * np.pi) * ellipse[1][0] / 2,
                                 center_in_cut[1] - np.sin(ellipse[2] / 180 * np.pi) * ellipse[1][0] / 2]])
    image_center = max(imc.shape) / 2, max(imc.shape) / 2
    target_points = np.float32([image_center,
                                [image_center[0] + ellipse[1][1] / 2,
                                 image_center[1]],
                                [image_center[0],
                                 image_center[1] - ellipse[1][1] / 2]])
    return warpAffine(imc, getAffineTransform(source_points, target_points), (max(imc.shape), max(imc.shape)))

def is_type2_marker(imc, ellipse):
    # Check for Type 2: black center (2/3 diameter) with white ring
    center_mask = np.zeros(imc.shape, dtype=np.uint8)
    circle(center_mask, (int(imc.shape[0] / 2 * 16), int(imc.shape[1] / 2 * 16)),
           radius=int(ellipse[1][1] * 0.333 * 16), color=255, thickness=-1, lineType=LINE_AA, shift=4)  # 2/3 diameter black
    ring_mask = np.zeros(imc.shape, dtype=np.uint8)
    circle(ring_mask, (int(imc.shape[0] / 2 * 16), int(imc.shape[1] / 2 * 16)),
           radius=int(ellipse[1][1] / 2 * 16), color=255, thickness=-1, lineType=LINE_AA, shift=4)
    ring_mask = ring_mask - center_mask  # White ring area
    center_sum = np.sum(imc * (center_mask / 255))
    sum_of_center_mask = np.sum(center_mask / 255)
    avg_center_color = center_sum / sum_of_center_mask if sum_of_center_mask > 0 else 0
    ring_sum = np.sum(imc * (ring_mask / 255))
    sum_of_ring_mask = np.sum(ring_mask / 255)
    avg_ring_color = ring_sum / sum_of_ring_mask if sum_of_ring_mask > 0 else 0
    # Check if center is dark and ring is bright
    if avg_center_color < avg_ring_color:
        contrast = (avg_ring_color - avg_center_color) / 255
        return contrast > 0.2  # Adjustable contrast threshold
    return False

def is_type1_marker(imc, ellipse):
    # Check for Type 1: uniformly white
    mask = np.zeros(imc.shape, dtype=np.uint8)
    circle(mask, (int(imc.shape[0] / 2 * 16), int(imc.shape[1] / 2 * 16)),
           radius=int(ellipse[1][1] / 2 * 16), color=255, thickness=-1, lineType=LINE_AA, shift=4)
    marker_sum = np.sum(imc * (mask / 255))
    sum_of_mask = np.sum(mask / 255)
    avg_color = marker_sum / sum_of_mask if sum_of_mask > 0 else 0
    # Check if marker is mostly white
    return avg_color > 200  # Adjustable threshold for whiteness

def collect_points(target_size, data):
    canvas = np.zeros((target_size[0] * 2 + 20, len(data) * target_size[1], 3), dtype=np.uint8)
    for i, point_data in enumerate(data):
        spi = point_data["skewed_point"]
        w = min(int(spi.shape[1] * target_size[1] / spi.shape[0]), target_size[0])
        h = min(int(spi.shape[0] * target_size[0] / spi.shape[1]), target_size[1])
        spis = resize(spi, (w, h), interpolation=INTER_CUBIC)
        pis = resize(point_data["unskewed_point"], target_size, interpolation=INTER_CUBIC)
        canvas[(target_size[0] - h) // 2: (target_size[0] - h) // 2 + h,
               i * target_size[1] + (target_size[1] - w) // 2: i * target_size[1] + (target_size[1] - w) // 2 + w] = np.stack([spis, spis, spis], axis=2)
        canvas[target_size[0]: target_size[0] + target_size[0],
               i * target_size[1]: i * target_size[1] + target_size[1]] = np.stack([pis, pis, pis], axis=2)
        putText(canvas, point_data["marker_type"], (i * target_size[1] + 4, target_size[0] * 2 + 12),
                fontFace=FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 255), thickness=1, lineType=LINE_AA)
    return canvas