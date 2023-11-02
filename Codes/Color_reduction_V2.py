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

#%%
# Parameters
num_colors_max = 15

file_path = "../Images/Band/"
filename = "images_info.txt"
lines = []
with open(file_path + filename, 'r') as file:
    for line in file:
        lines.append(line)
opt_num_color=[]
goal_num_color=[]

for i in range(0, 10):
    n = i+30
    imagename = f'Test_images_{n}'
    
    # Find the number of colors goal
    for line in lines:
        if imagename in line:
            goal_num_color.append(int(line.split(";")[2]))
            break
    
    # Load the image
    image = np.array(Image.open(file_path + imagename + ".png"))
    
    # Making sure that there is no "transparency" channel
    if image.shape[-1] == 4:
        image = image[..., :3]
        
    #Convert to hsv
    hsv_image = color.rgb2hsv(image)
        
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    hsv_pixels = hsv_image.reshape(-1, 3)
    
    HSV_images, HSV_colors, HSV_error_tab = quantized_colors_with_Kmeans(pixels, 'HSV', image, num_colors_max, hsv_pixels)

    opt_num_color.append(find_optimal_color_number(HSV_error_tab[0]))
    
    if opt_num_color[i] != goal_num_color[i]:
        display_images(image, HSV_images, num_colors_max) 
        error_plot(HSV_error_tab[0], num_colors_max, "RGB space")
        error_plot(HSV_error_tab[1], num_colors_max, "HSV space")
        
    
print(opt_num_color)
print(goal_num_color)