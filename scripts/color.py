import cv2
import numpy as np
import skimage.feature as feature
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial.distance import cityblock
import scipy.io

import warnings
warnings.filterwarnings("ignore")

# Load w2c matrix from .mat file
w2c_data = scipy.io.loadmat('w2c.mat')
w2c = w2c_data['w2c']


def color_features(frame, w2c=w2c):    
    round_gidits = 4
    
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # Calculate the brightness, hue, and saturation features
    brightness = np.mean(hsv[:, :, 2]).round(round_gidits)  # Mean of the value (V) channel
    hue = np.mean(hsv[:, :, 0]).round(3)  # Mean of the hue (H) channel
    saturation = np.mean(hsv[:, :, 1]).round(round_gidits)  # Mean of the saturation (S) channel
    
    # Calculate the brightness contrast
    brightness_contrast = np.std(hsv[:, :, 2]).round(round_gidits)  # Standard deviation of the value (V) channel
    
    # Calculate the color diversity using the Earth Mover's Distance (EMD)
    # between the frame's color histogram and a uniform histogram
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    uniform_hist = np.ones_like(hist) / (180 * 256)
    color_diversity = cityblock(hist.flatten(), uniform_hist.flatten()).round(round_gidits)
    
    # Calculate the color amounts using the im2c() function
    color_amounts = {}
    total_pixels = frame.shape[0] * frame.shape[1]

    color_names = ['color_'+i for i in ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']]
    pixel_color_name = np.array(im2c(frame, w2c, 0)) 
    for i,color_name in enumerate(color_names):
        color_amounts[color_name] = (np.sum(pixel_color_name==i)/total_pixels).round(3)
    
    # Compute clarity of colors and their intensity
    hue_std = np.std(hsv[:, :, 0])
    sat_std = np.std(hsv[:, :, 1])
    val_std = np.std(hsv[:, :, 2])
    clarity = ((hue_std + sat_std + val_std) / 3).round(round_gidits)
    
    return {
        'color_brightness': brightness,
        'color_hue': hue,
        'color_saturation': saturation,
        'color_brightness_contrast': brightness_contrast,
        'color_color_diversity': color_diversity,
        'color_clarity': clarity,
         **color_amounts,
    }


def im2c(im, w2c, color=0):
    # input im should be DOUBLE !
    # color=0 is color names out
    # color=-1 is colored frame with color names out
    # color=1-11 is prob of colorname=color out;

    # Convert the input frame to double precision
    im = im.astype(np.float64)
    
    # order of color names: black,blue, brown, grey, green, orange, pink, purple, red, white, yellow
    color_values = np.array([
        [0, 0, 0], [0, 0, 1], [0.5, 0.4, 0.25], [0.5, 0.5, 0.5], [0, 1, 0],
        [1, 0.8, 0], [1, 0.5, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0]
    ])
    
    # Calculate the index frame
    RR, GG, BB = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    im = cv2.merge((RR, GG, BB))
    index_im = 1 + np.floor(RR / 8) + 32 * np.floor(GG / 8) + 32 * 32 * np.floor(BB / 8)
    index_im = [i-1 for i in index_im .astype(int)]
    
    if color == 0:
        max1, w2cM = np.max(w2c, axis=1), np.argmax(w2c, axis=1)
        return w2cM[index_im]
    
    if color > 0 and color < 12:
        w2cM = w2c[:, color - 1]
        return w2cM[index_im.astype(int).flatten()].reshape(im.shape[0], im.shape[1])
    
    if color == -1:
        out = im.copy()
        max1, w2cM = np.max(w2c, axis=1), np.argmax(w2c, axis=1)
        out2 = w2cM[index_im.astype(int).flatten()].reshape(im.shape[0], im.shape[1])
        
        for jj in range(im.shape[0]):
            for ii in range(im.shape[1]):
                out[jj, ii, :] = color_values[out2[jj, ii]] * 255
        
        return out