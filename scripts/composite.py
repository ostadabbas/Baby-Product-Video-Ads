from math import sqrt
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv


def analyze_rule_of_thirds(image: str, bbox: tuple, form: str = 'xyxy') -> dict:
    """
    Analyzes the placement of an object within a rule of thirds grid.
    
    Args:
        image (str): The path to the input image.
        bbox (tuple): The bounding box of the object in (x, y, w, h) format.
    
    Returns:
        dict: A dictionary containing the following keys:
            - 'on_lines': Boolean indicating if the object is placed on the rule of thirds lines.
            - 'at_intersections': Boolean indicating if the object is placed at the rule of thirds intersections.
            - 'distances': A dictionary containing the normalized distances from the object center to the four corners of the rule of thirds grid.
    """
    # Get the dimensions of the image
    height, width, _ = image.shape
    
    # Calculate the rule of thirds grid lines
    third_width = width / 3
    third_height = height / 3
    
    if form == 'xyxy':
        x, y, x1, y1 = bbox
        w = x1 - x
        h = y1 - y
    else:
        # Extract the bounding box coordinates
        x, y, w, h = bbox
    
    # Normalize the bounding box coordinates and dimensions
    norm_x = x / width
    norm_y = y / height
    norm_w = w / width
    norm_h = h / height
    
    # Calculate the center of the normalized bounding box
    norm_box_center_x = norm_x + norm_w / 2
    norm_box_center_y = norm_y + norm_h / 2
    
    # Calculate the normalized distances to the four corners of the rule of thirds grid
    norm_dist_top_left = sqrt((norm_box_center_x - 1/3) ** 2 + (norm_box_center_y - 1/3) ** 2)
    norm_dist_top_right = sqrt((norm_box_center_x - 2/3) ** 2 + (norm_box_center_y - 1/3) ** 2)
    norm_dist_bottom_left = sqrt((norm_box_center_x - 1/3) ** 2 + (norm_box_center_y - 2/3) ** 2)
    norm_dist_bottom_right = sqrt((norm_box_center_x - 2/3) ** 2 + (norm_box_center_y - 2/3) ** 2)
    
    # Check if the object is placed on the rule of thirds lines
    on_lines = (norm_x <= 1/3 or norm_x + norm_w >= 2/3 or
                norm_y <= 1/3 or norm_y + norm_h >= 2/3)
    
    # Check if the object is placed at the rule of thirds intersections
    at_intersections = (norm_x <= 1/3 and norm_y <= 1/3) or \
                       (norm_x <= 1/3 and norm_y + norm_h >= 2/3) or \
                       (norm_x + norm_w >= 2/3 and norm_y <= 1/3) or \
                       (norm_x + norm_w >= 2/3 and norm_y + norm_h >= 2/3)
    
    return {
        #'on_lines': on_lines,
        'rot_at_intersections': at_intersections,
        'rot_dist_top_left': round(norm_dist_top_left,4),
        'rot_dist_top_right': round(norm_dist_top_right,4),
        'rot_dist_bottom_left': round(norm_dist_bottom_left,4),
        'rot_dist_bottom_right': round(norm_dist_bottom_right,4)
        }

def inner_rectangle_features(image, visualize=False):
    # Convert the image to HSV color space
    hsv = rgb2hsv(image)
    
    # Get the dimensions of the image
    image_height, image_width, _ = image.shape
    
    # Calculate the rule of thirds grid lines
    third_width = image_width / 3
    third_height = image_height / 3
    
    # Calculate the inner rectangle coordinates
    inner_x1 = third_width
    inner_y1 = third_height
    inner_x2 = 2 * third_width
    inner_y2 = 2 * third_height
    
    # Extract the pixels inside the inner rectangle from the V channel
    inner_v_values = hsv[int(inner_y1):int(inner_y2), int(inner_x1):int(inner_x2), 2]
    inner_brightness = np.mean(inner_v_values)
    total_brightness = np.mean(hsv[:, :, 2])
    brightness_ratio = inner_brightness / total_brightness
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Compute the local mean of the grayscale image using a sliding window
    kernel_size = 3
    local_mean = cv2.blur(gray, (kernel_size, kernel_size))
    # Compute the sharpness as the sum of the absolute values of the Laplacian divided by the local mean
    total_sharpness = np.sum(np.abs(lap)) / np.sum(local_mean)
    
    # Compute the sharpness of the inner rectangle
    inner_lap = cv2.Laplacian(inner_v_values.astype(np.uint8), cv2.CV_64F)
    inner_local_mean = cv2.blur(inner_v_values.astype(np.uint8), (kernel_size, kernel_size))
    inner_sharpness = np.sum(np.abs(inner_lap)) / np.sum(inner_local_mean)
    
    # Calculate the sharpness ratio
    if np.sum(inner_local_mean) == 0:
        sharpness_ratio = 0
    else:
        sharpness_ratio = inner_sharpness / total_sharpness

    if visualize:
        # Plot the image
        plt.imshow(image)

        # Plot the inner rectangle in red
        plt.plot([inner_x1, inner_x2, inner_x2, inner_x1, inner_x1], 
                 [inner_y1, inner_y1, inner_y2, inner_y2, inner_y1], 'r-')

        # Fill the inner rectangle with red color
        plt.fill([inner_x1, inner_x2, inner_x2, inner_x1, inner_x1], 
                 [inner_y1, inner_y1, inner_y2, inner_y2, inner_y1], 'r', alpha=0.2)

        # Draw the vertical grid lines
        plt.axvline(x=third_width, color='g', linestyle='--')
        plt.axvline(x=2 * third_width, color='g', linestyle='--')

        # Draw the horizontal grid lines
        plt.axhline(y=third_height, color='g', linestyle='--')
        plt.axhline(y=2 * third_height, color='g', linestyle='--')
        
        plt.show()
    
    return {'inner_brightness': round(brightness_ratio,4), 'inner_sharpness':round(sharpness_ratio,4)}