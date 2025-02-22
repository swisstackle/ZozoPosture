from __future__ import division, print_function, unicode_literals
from markers import find_marker_ellipses, unskew_point, is_type1_marker, is_type2_marker, collect_points
import cv2
import sys
import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict
from scipy.interpolate import splrep, splev

def scale_preview(preview_img, max_preview_size=(1800, 960)):
    scale_fac = min(max_preview_size[1] / preview_img.shape[0], max_preview_size[0] / preview_img.shape[1], 1)
    scaled_im = cv2.resize(preview_img, (int(preview_img.shape[1] * scale_fac), int(preview_img.shape[0] * scale_fac)),
                           interpolation=cv2.INTER_CUBIC)
    return scaled_im, scale_fac

def filter_isolated_markers(positions, radius=10, amount=2):
    """
    Filters out markers with no neighbors within the specified radius using a KD-tree.
    
    Args:
        positions: List of (x, y) coordinates of detected markers.
        radius: Distance threshold in pixels (default: 10).
        amount: Minimum number of neighbors required (default: 2).
    
    Returns:
        List of indices of non-isolated markers.
    """
    if len(positions) < 2:
        return []  # If there’s only one or no markers, consider them isolated

    tree = KDTree(positions)
    neighbors = tree.query_ball_point(positions, r=radius, p=2, return_sorted=False)
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
    filtered_indices = filter_isolated_markers(positions, radius=30, amount=7)
    
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

    # Step 1: Group markers by row (y-coordinate) and find leftmost markers
    markers_by_row = defaultdict(list)
    for i in range(len(positions)):
        y = int(positions[i][1])  # y-coordinate
        x = int(positions[i][0])  # x-coordinate
        markers_by_row[y].append(x)

    # Find the leftmost marker (smallest x-coordinate) for each row
    leftmost_markers = []
    for y, xs in markers_by_row.items():
        min_x = min(xs)  # Smallest x is the leftmost marker
        leftmost_markers.append((min_x, y))

    # Convert leftmost markers to numpy array for processing
    positions = np.array(leftmost_markers)

    # Step 2: Identify the region of interest
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    bin_size = 15  # Adjust bin size as needed
    bins = np.arange(y_min, y_max + bin_size, bin_size)
    back_points = []
    for y_start in bins:
        y_end = y_start + bin_size
        bin_points = positions[(positions[:, 1] >= y_start) & (positions[:, 1] < y_end)]
        if len(bin_points) > 0:
            leftmost = bin_points[np.argmin(bin_points[:, 0])]
            back_points.append(leftmost)

    back_points = np.array(back_points)

    # Step 3: Fit an initial polynomial curve
    if len(back_points) > 3:
        y_values = back_points[:, 1]
        x_values = back_points[:, 0]
        degree = 3  # Cubic polynomial
        coefficients = np.polyfit(y_values, x_values, degree)
        x_pred = np.polyval(coefficients, y_values)

        # Step 4: Filter outliers
        residuals = x_values - x_pred
        threshold = 1 * np.std(residuals)
        inliers = np.abs(residuals) < threshold
        filtered_positions = back_points[inliers]

        # Step 5: Fit a refined spline curve
        if len(filtered_positions) > 3:
            y_smooth = np.linspace(filtered_positions[:, 1].min(), filtered_positions[:, 1].max(), 100)
            spline = splrep(filtered_positions[:, 1], filtered_positions[:, 0], s=250)
            x_smooth = splev(y_smooth, spline)

            # Create a blank image for the back outline
            back_outline_img = np.zeros_like(im)

            # Draw the smoothed back outline
            for i in range(len(y_smooth) - 1):
                cv2.line(back_outline_img, 
                         (int(x_smooth[i]), int(y_smooth[i])), 
                         (int(x_smooth[i + 1]), int(y_smooth[i + 1])), 
                         color=(0, 255, 0), thickness=3)

            # Save and display the back outline
            cv2.imwrite("back_outline.png", back_outline_img)
            cv2.imshow('Back Outline', back_outline_img)
            cv2.waitKey(0)

            # Step 6: Analyze Spine Curvature and Alignment
            # Compute first and second derivatives using finite differences
            x_prime = np.gradient(x_smooth, y_smooth)
            x_double_prime = np.gradient(x_prime, y_smooth)
            y_prime = np.gradient(np.ones_like(y_smooth), y_smooth)  # y is linear, so y' = 1
            y_double_prime = np.gradient(y_prime, y_smooth)  # y'' = 0, but included for consistency

            print(f'x_prime: {x_prime}')
            print(f'x_double_prime: {x_double_prime}')

            # Calculate curvature at each point
            curvature = np.abs(x_prime * y_double_prime - y_prime * x_double_prime) / (x_prime**2 + y_prime**2)**1.5
            curvature = np.nan_to_num(curvature, nan=0.0)  # Handle any NaN values

            print(curvature)

            # Step 7: Analyze Upper and Lower Spine Curvature
            y_mid = np.median(y_smooth)
            upper_mask = y_smooth <= y_mid
            lower_mask = y_smooth > y_mid

            print(upper_mask)
            print(lower_mask)
            print(np.mean(curvature[upper_mask]))

            curvature_upper = np.mean(curvature[upper_mask]) if np.any(upper_mask) else 0.0
            curvature_lower = np.mean(curvature[lower_mask]) if np.any(lower_mask) else 0.0

            # Step 8: Assess Overall Alignment
            # Fit a linear regression to get the slope
            slope, _ = np.polyfit(y_smooth, x_smooth, 1)
            
            # Calculate centroid for additional alignment insight
            x_centroid = np.mean(x_smooth)
            y_centroid = np.mean(y_smooth)
            image_center_x = im.shape[1] / 2  # Center of the image horizontally
            lean_offset = x_centroid - image_center_x  # Positive = lean right, negative = lean left

            # Output analysis
            print(f"Upper Spine Curvature: {curvature_upper:.4f}")
            print(f"Lower Spine Curvature: {curvature_lower:.4f}")
            print(f"Overall Spine Slope (Alignment): {slope:.4f} (Positive = lean right, Negative = lean left)")
            print(f"Spine Centroid Lean Offset: {lean_offset:.2f} pixels (relative to image center)")

        else:
            print("Not enough points to fit a spline for the back outline.")
    else:
        print("Not enough points to fit an initial curve.")

    cv2.destroyAllWindows()