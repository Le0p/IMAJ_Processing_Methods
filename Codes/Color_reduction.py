# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 12:03:25 2023

@author: Leonard TREIL
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from skimage import color

def display_images(image, quantized_images_tab, num_color_max):
    
    # Display the original and quantized images
    plt.figure(figsize=(15, 25))
    plt.subplot(num_colors_max//5, 5, 1)
    plt.imshow(image)
    plt.title('Original Image')

    for num_colors in range(1, num_colors_max):        
        plt.subplot(num_colors_max//5, 5, num_colors+1)
        plt.imshow(quantized_images_tab[num_colors-1])
        plt.title(f'Quantized Image ({num_colors} Colors)')

    plt.tight_layout()
    plt.show()
    
def error_plot(error_tab, num_color_max, title=None):
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1,num_colors_max), error_tab)
    plt.xlabel("Number of clusters")
    plt.ylabel("MSE Error")
    if title != None:
        plt.title(title)
    plt.show()
    
def color_report(quantized_colors_tab, num_colors_max):
    for num_colors in range(1, num_colors_max):
        print(f"RGB report ({num_colors} Colors): ")
        print(quantized_colors_tab[num_colors-1])

def quantized_colors_with_Kmeans(pixels, pixel_type, image, num_colors_max, pixels_HSV = None):
    
    quantized_colors_tab = []
    quantized_images_tab = []
    mse_tab = []
    mse_RGB_tab = []
    mse_HSV_tab = []

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
        mse_RGB = mean_squared_error(pixels, quantized_colors[labels])/(255**2)
        mse_RGB_tab.append(mse_RGB)
        if pixel_type == 'HSV':
            mse_HSV = mean_squared_error(pixels_HSV, kmeans.cluster_centers_[labels])
            mse_HSV_tab.append(mse_HSV)
            
        quantized_colors_tab.append(quantized_colors)
        quantized_images_tab.append(quantized_image)
    
    mse_tab.append(mse_RGB_tab)
    if pixel_type == 'HSV':
        mse_tab.append(mse_HSV_tab)
        
    display_images(image, quantized_images_tab, num_colors_max) 
    error_plot(mse_RGB_tab, num_colors_max, "RGB space")
    if pixel_type == 'HSV':
        error_plot(mse_HSV_tab, num_colors_max, "HSV space")
    
    return quantized_images_tab, quantized_colors_tab, mse_tab

# Parameters
image_path = 'IMG_test.png'
num_colors_max = 20

# Load the image
image = np.array(Image.open(image_path))

# Making sure that there is no "transparency" channel
if image.shape[-1] == 4:
    image = image[..., :3]
    
#Convert to hsv
hsv_image = color.rgb2hsv(image)
    
# Reshape the image to a 2D array of pixels
pixels = image.reshape(-1, 3)
hsv_pixels = hsv_image.reshape(-1, 3)

#RGB_images, RGB_colors, RGB_error_tab = quantized_colors_with_Kmeans(pixels, 'RGB', image, num_colors_max)  
HSV_images, HSV_colors, HSV_error_tab = quantized_colors_with_Kmeans(pixels, 'HSV', image, num_colors_max, hsv_pixels)

