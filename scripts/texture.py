import cv2
import numpy as np
import skimage.feature as feature

    
def texture_features(frame):
    """
    Extract GLCM features of contrast, correlation, energy, and homogeneity for each HSV channel.
    
    Args:
        frame (numpy.ndarray): Input frame.
        
    Returns:
        dict: A dictionary containing the GLCM features for each HSV channel.
    """
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2HSV)
    
    # Extract GLCM features for each HSV channel
    features = {}
    for channel in range(hsv_frame.shape[2]):
        gray_frame = hsv_frame[:, :, channel]
        
        # Calculate the GLCM matrix
        glcm = feature.graycomatrix(gray_frame, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
        
        # Extract the GLCM features
        contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
        correlation = feature.graycoprops(glcm, 'correlation')[0, 0]
        energy = feature.graycoprops(glcm, 'energy')[0, 0]
        homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
        dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
        
        features[f'texture_c{channel+1}_contrast'] = round(contrast,4)
        features[f'texture_c{channel+1}_correlation'] = round(correlation,4)
        features[f'texture_c{channel+1}_energy'] = round(energy,4)
        features[f'texture_c{channel+1}_homogeneity'] = round(homogeneity,4)
        features[f'texture_c{channel+1}_dissimilarity'] = round(dissimilarity,4)
        
    
    return features