# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:18:30 2023

@author: Léoanrd TREIL
"""

import pandas as pd
import os
from skimage import io, color
from skimage.filters import meijering, sato, frangi, hessian
import matplotlib.pyplot as plt
import numpy as np

# Function to find a file by name in a directory and its subdirectories
def find_file(start_dir, filename):
    for dirpath, dirnames, filenames in os.walk(start_dir):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None  # File not found

def original(image, **kwargs):
    """Return the original image, ignoring any kwargs."""
    return image

def extract_image_list(folder_path):
    
    image_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_list.append(filename)
    
    sorted(image_list)
    
    return image_list

folder_path = "E:/Images_IMAJ"
folder_path = "C:/Users/Hp/Nextcloud/EPFL/Master 3/IMAJ_Processing_Methods/Images/IMG_maj"

# Load the DataFrame from the CSV file
df = pd.read_csv("../Images/IMAJ_image_database.csv")

random_samples = df.sample(n=1)
random_samples = extract_image_list(folder_path)

sum_results = []

for sample in random_samples:
    print(sample)
    
    #image_name = sample['Filename']
    image_name = sample
    image_path = find_file(folder_path, image_name)
    
    # Load the image
    image = io.imread(image_path)
    
    # Convert the image to grayscale
    image_gray = color.rgb2gray(image)
    cmap = plt.cm.gray
    
    temp_results = []
    
    fig, axs = plt.subplots(2, 3, figsize=(15,10))
    for i, black_ridges in enumerate([True, False]):
        for j, (func, sigmas) in enumerate([
                (original, None),
                (meijering, [1]),
                (meijering, range(1, 5))
                ]):
            result = func(image_gray, black_ridges=black_ridges, sigmas=sigmas)
            axs[i, j].imshow(result, cmap=cmap)
            if i == 0:
                title = func.__name__
                if sigmas:
                    title += f"\n\N{GREEK SMALL LETTER SIGMA} = {list(sigmas)}"
                axs[i, j].set_title(title)
            if j == 0:
                axs[i, j].set_ylabel(f'{black_ridges = }')
            else:
                norm_sum = np.sum(np.sqrt(result))/(np.shape(image_gray)[0]*np.shape(image_gray)[1])
                temp_results.append(norm_sum)
                axs[i, j].set_ylabel(f'norm_sum = {norm_sum}')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            
    sum_results.append(temp_results)
    
    plt.tight_layout()
    plt.show()
    break

sum_results=np.array(sum_results)

#%%
print("Black Ridge = True and meijering = [1]:", sum_results[:, 0])
print("Black Ridge = True and meijering = [1,2,3,4]:", sum_results[:, 1])
print("Black Ridge = False and meijering = [1]:", sum_results[:, 2])
print("Black Ridge = False and meijering = [1,2,3,4]:", sum_results[:, 3])
    