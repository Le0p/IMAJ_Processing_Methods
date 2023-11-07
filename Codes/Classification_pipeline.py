# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:26:21 2023

@author: Léoanrd TREIL
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage import color
import Color_library

def extract_image_list(folder_path):
    
    image_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_list.append(filename)
    
    sorted(image_list)
    
    return image_list
    
def load_image(image_path):
    
    # Load the image
    image = np.array(Image.open(image_path))
    
    # Making sure that there is no "transparency" channel
    if image.shape[-1] == 4:
        image = image[..., :3]
        
    return image

def extract_KMeans_colors_features(image):
    
    # Vecteur features -> nb classes, 3 couleurs dominantes (RGB) 
    # + %remplissage essayé avec HSV aussi
    
    num_colors_max = 15
    
    #Convert to hsv
    hsv_image = color.rgb2hsv(image)
    
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    hsv_pixels = hsv_image.reshape(-1, 3)
    
    HSV_images, HSV_colors, HSV_error_tab = Color_library.quantized_colors_with_Kmeans(pixels, 'HSV', image, num_colors_max, hsv_pixels)

    opt_num_color = Color_library.find_optimal_color_number(HSV_error_tab[0])
    
    image_features = [opt_num_color]
    
    
    
    for i in range(3):
        image_features.append(HSV_colors[opt_num_color - 1][i][0])
        image_features.append(HSV_colors[opt_num_color - 1][i][1])
        image_features.append(HSV_colors[opt_num_color - 1][i][2])
        image_features.append(np.mean(sum(sum(HSV_images[opt_num_color] == HSV_colors[opt_num_color - 1][i]))/len(pixels)))
    
    return image_features



folder_path = "../Images/Band/"

image_list = extract_image_list(folder_path)

for image_name in image_list:
    
    image = load_image(folder_path + image_name)
    
    image_features = extract_KMeans_colors_features(image)
    print(image_features)