# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:32:00 2023

@author: Léonard TREIL
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from skimage import color

def quantized_colors_with_Kmeans(pixels, pixel_type, image, num_colors_max, pixels_HSV = None):
    
    quantized_colors_tab = []
    quantized_images_tab = []
    nmse_tab = []
    nmse_RGB_tab = []
    nmse_HSV_tab = []

    for num_colors in tqdm(range(1, num_colors_max)):

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init='auto')
        if pixel_type == 'RGB':
            kmeans.fit(pixels)
            quantized_colors = np.round(kmeans.cluster_centers_).astype(int)     
        elif pixel_type == 'HSV':
            kmeans.fit(pixels_HSV)
            quantized_colors = (color.hsv2rgb(kmeans.cluster_centers_)*255).astype(int)
        labels = kmeans.labels_
        
        # Replace each pixel with its corresponding quantized color
        quantized_image = quantized_colors[labels].reshape(image.shape)
        
        # Compute error
        nmse_RGB = 1 - mean_squared_error(pixels, quantized_colors[labels])/np.var(pixels)
        nmse_RGB_tab.append(nmse_RGB)
        if pixel_type == 'HSV':
            nmse_HSV = 1 - mean_squared_error(pixels_HSV, kmeans.cluster_centers_[labels])/np.var(pixels_HSV)
            nmse_HSV_tab.append(nmse_HSV)
            
        quantized_colors_tab.append(quantized_colors)
        quantized_images_tab.append(quantized_image)
    
    nmse_tab.append(nmse_RGB_tab)
    if pixel_type == 'HSV':
        nmse_tab.append(nmse_HSV_tab)
        
    #display_images(image, quantized_images_tab, num_colors_max) 
    #error_plot(nmse_RGB_tab, num_colors_max, "RGB space")
    #if pixel_type == 'HSV':
    #    error_plot(nmse_HSV_tab, num_colors_max, "HSV space")
    
    return quantized_images_tab, quantized_colors_tab, nmse_tab

def find_optimal_color_number(error_tab):
    number_colors = 0
    same_value_count = 0
    last_value = 0
    
    for i in range(2,len(error_tab)):
        
        # Check for a plateau to take the first value of the plateau
        if error_tab[i] == last_value:
            same_value_count += 1
        else:
            last_value = error_tab[i]
            same_value_count = 0
        
        # Simple thrshold
        if error_tab[i] > 0.95:
            break
    
    number_colors = i + 1 - same_value_count
    
    return number_colors