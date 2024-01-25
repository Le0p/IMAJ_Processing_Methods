# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:58:37 2024

@author: Leonard TREIL
"""

import os
import numpy as np
from PIL import Image

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

# Function to find a file by name in a directory and its subdirectories
def find_file(start_dir, filename):
    for dirpath, dirnames, filenames in os.walk(start_dir):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None  # File not found

