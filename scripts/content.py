from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
from math import sqrt
import numpy as np


mtcnn = MTCNN(keep_all=True)

def face_count_mtcnn(frame):
    results = {}
    boxes, _, _ = mtcnn.detect(frame, landmarks=True) #xyxy
    
    if np.any(boxes):
        sorted_boxes = sorted(boxes, key=compute_area, reverse=True)
        #only conside the rule of third of the largest area face
        rule_of_third_face = analyze_rule_of_thirds(frame, sorted_boxes[0])
        results.update({'face_count':boxes.shape[0]})
        results.update(rule_of_third_face)
    else:
        results = {'face_count':0,
            'rot_at_intersections': 0,
        'rot_dist_top_left': 2,
        'rot_dist_top_right': 2,
        'rot_dist_bottom_left': 2,
        'rot_dist_bottom_right': 2}
    
    return results

def compute_area(box):
    x1, y1, x2, y2 = box
    
    return (x2 - x1) * (y2 - y1)


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

def get_sam_stats(anns, image_size):
    if len(anns) == 0:
        return 0, 0
    
    # Get the areas of all the masks
    areas = [ann['area']/image_size for ann in anns if ann['area'] > image_size/100]
    avg_area = np.mean(areas)
    num_regions = len(areas)
    
    areas_all = [ann['area']/image_size for ann in anns]
    
    return {'region_size_avg': round(avg_area,4), 'region_count': num_regions, 'region_size_avg_all': round(np.mean(areas_all),4), 'region_count_all':len(areas_all)}