# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:45:43 2023

@author: Leonard TREIL
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform

# Function to find a file by name in a directory and its subdirectories
def find_file(start_dir, filename):
    for dirpath, dirnames, filenames in os.walk(start_dir):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None  # File not found

folder_path = "E:/Images_IMAJ"

# Load the DataFrame from the CSV file
df = pd.read_csv("../Images/IMAJ_image_database.csv")

random_samples = df.sample(n=1)

for sample in random_samples.iloc():
    print(sample)
    
    image_name = sample['Filename']
    image_path = find_file(folder_path, image_name)

    # Load the image
    image = io.imread(image_path)

    # Set up figure for plotting
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    # Display original image
    axes[0].imshow(image)
    axes[0].set_title(f'image_size:{image.shape}')
    axes[0].axis('off')
    
    # Rescale the image down 10 times and display in the figure
    for i in range(1, 10):
        # Reduce image size by a factor of 10 each time
        scale_factor = i / 10
        resized_image = transform.rescale(image, scale_factor, anti_aliasing=True, channel_axis=2)
        
        # Display the rescaled image
        axes[i].imshow(resized_image)
        axes[i].set_title(f'image_size:{resized_image.shape}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
