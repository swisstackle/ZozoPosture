#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from markers import find_marker_ellipses, unskew_point, is_type1_marker, is_type2_marker, collect_points
import cv2
import sys

def scale_preview(preview_img, max_preview_size=(1800, 960)):
    scale_fac = min(max_preview_size[1] / preview_img.shape[0], max_preview_size[0] / preview_img.shape[1], 1)
    scaled_im = cv2.resize(preview_img, (int(preview_img.shape[1] * scale_fac), int(preview_img.shape[0] * scale_fac)),
                           interpolation=cv2.INTER_CUBIC)
    return scaled_im, scale_fac

def detect_points(img):
    skewed_points, origins, ellipses = find_marker_ellipses(img)
    unskewed_points = [unskew_point(skewed_points[i], origins[i], ellipses[i]) for i in range(len(skewed_points))]
    marker_types = []
    positions = []
    for i in range(len(unskewed_points)):
        positions.append(ellipses[i][0])
        if is_type2_marker(unskewed_points[i], ellipses[i]):
            marker_types.append("Type2")
        elif is_type1_marker(unskewed_points[i], ellipses[i]):
            marker_types.append("Type1")
        else:
            marker_types.append("Unknown")
    raw_data = [{
        "skewed_point": skewed_points[i],
        "unskewed_point": unskewed_points[i],
        "origin": origins[i],
        "ellipse": ellipses[i],
        "marker_type": marker_types[i],
        "position": positions[i]
    } for i in range(len(unskewed_points))]
    return marker_types, positions, raw_data

if __name__ == '__main__':
    args = sys.argv
    input_file_name = args[1]
    im = cv2.imread(input_file_name)
    marker_types, positions, raw_data = detect_points(im)

    p_coll_img = collect_points((64, 64), raw_data)
    if min(p_coll_img.shape[0:1]) > 0:
        cv2.imwrite("collected_points.png", p_coll_img)

    im_draw = im.copy()
    for i in range(len(marker_types)):
        # Draw the ellipses for the detected markers
        cv2.ellipse(im_draw, raw_data[i]["ellipse"], color=(0, 255, 0), thickness=3)
        # The line below that adds labels has been removed to keep the image clear
        # cv2.putText(im_draw, raw_data[i]["marker_type"], (int(positions[i][0]), int(positions[i][1])),
        #             fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # Save the image without labels
    cv2.imwrite("point_positions.png", im_draw)
    im_preview, scale_factor = scale_preview(im_draw)
    cv2.imshow('Point positions', im_preview)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()