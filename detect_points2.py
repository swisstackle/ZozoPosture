from __future__ import division, print_function, unicode_literals
from markers import find_marker_ellipses, unskew_point, is_type1_marker, is_type2_marker, collect_points
import cv2
import sys
import numpy as np
from scipy.spatial import KDTree

def scale_preview(preview_img, max_preview_size=(1800, 960)):
    scale_fac = min(max_preview_size[1] / preview_img.shape[0], max_preview_size[0] / preview_img.shape[1], 1)
    scaled_im = cv2.resize(preview_img, (int(preview_img.shape[1] * scale_fac), int(preview_img.shape[0] * scale_fac)),
                           interpolation=cv2.INTER_CUBIC)
    return scaled_im, scale_fac

def filter_isolated_markers(positions, radius=10, amount = 2):
    """
    Filters out markers with no neighbors within the specified radius using a KD-tree.
    
    Args:
        positions: List of (x, y) coordinates of detected markers.
        radius: Distance threshold in pixels (default: 40).
    
    Returns:
        List of indices of non-isolated markers.
    """
    if len(positions) < 2:
        return []  # If there’s only one or no markers, consider them isolated

    # Create a KD-tree from the marker positions
    tree = KDTree(positions)
    
    # Find neighbors within the radius (includes the point itself)
    neighbors = tree.query_ball_point(positions, r=radius, p=2, return_sorted=False)
    
    # Keep markers with at least one neighbor (len > 1 because it counts itself)
    filtered_indices = [i for i, n in enumerate(neighbors) if len(n) > amount]
    
    return filtered_indices

def detect_points(img):
    # Detect markers and extract their properties
    skewed_points, origins, ellipses = find_marker_ellipses(img)
    unskewed_points = [unskew_point(skewed_points[i], origins[i], ellipses[i]) for i in range(len(skewed_points))]
    
    marker_types = []
    positions = []
    for i in range(len(unskewed_points)):
        positions.append(ellipses[i][0])  # Center of the ellipse
        if is_type2_marker(unskewed_points[i], ellipses[i]):
            marker_types.append("Type2")
        elif is_type1_marker(unskewed_points[i], ellipses[i]):
            marker_types.append("Type1")
        else:
            marker_types.append("Unknown")
    
    # Filter out isolated markers
    filtered_indices = filter_isolated_markers(positions, radius=30, amount = 7)
    
    # Retain only non-isolated markers
    filtered_marker_types = [marker_types[i] for i in filtered_indices]
    filtered_positions = [positions[i] for i in filtered_indices]
    raw_data = [{
        "skewed_point": skewed_points[i],
        "unskewed_point": unskewed_points[i],
        "origin": origins[i],
        "ellipse": ellipses[i],
        "marker_type": marker_types[i],
        "position": positions[i]
    } for i in filtered_indices]  # Only filtered markers in raw_data
    
    return filtered_marker_types, filtered_positions, raw_data

if __name__ == '__main__':
    args = sys.argv
    input_file_name = args[1]
    im = cv2.imread(input_file_name)
    marker_types, positions, raw_data = detect_points(im)

    # Generate collected points image
    p_coll_img = collect_points((64, 64), raw_data)
    if min(p_coll_img.shape[0:1]) > 0:
        cv2.imwrite("collected_points.png", p_coll_img)

    # Draw detected markers on the image
    im_draw = im.copy()
    for i in range(len(marker_types)):
        cv2.ellipse(im_draw, raw_data[i]["ellipse"], color=(0, 255, 0), thickness=3)

    # Save and display the result
    cv2.imwrite("point_positions.png", im_draw)
    im_preview, scale_factor = scale_preview(im_draw)
    cv2.imshow('Point positions', im_preview)
    key = cv2.waitKey(0)
        # Create a blank canvas with the same dimensions as the original image
    marker_only_img = np.zeros_like(im)
    
    # Draw only the filtered markers on the blank canvas
    for i in range(len(marker_types)):
        cv2.ellipse(marker_only_img, raw_data[i]["ellipse"], color=(0, 255, 0), thickness=3)
    
    # Save and display the marker-only image
    cv2.imwrite("markers_only.png", marker_only_img)
    cv2.imshow('Markers Only', scale_preview(marker_only_img)[0])
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()