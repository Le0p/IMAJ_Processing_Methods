# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:26:21 2023

@author: Léoanrd TREIL
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage import color
import Color_library


def load_image(image_path):
    
    # Load the image
    image = np.array(Image.open(image_path))
    
    # Making sure that there is no "transparency" channel
    if image.shape[-1] == 4:
        image = image[..., :3]

def extract_KMeans_colors(image):
    
    num_colors_max = 15
    
    #Convert to hsv
    hsv_image = color.rgb2hsv(image)
    
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    hsv_pixels = hsv_image.reshape(-1, 3)
    
    HSV_images, HSV_colors, HSV_error_tab = Color_library.quantized_colors_with_Kmeans(pixels, 'HSV', image, num_colors_max, hsv_pixels)

    opt_num_color = Color_library.find_optimal_color_number(HSV_error_tab[0])
    
    return opt_num_color, HSV_colors

