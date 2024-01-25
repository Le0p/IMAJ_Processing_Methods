# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:24:33 2023

@author: Léonard TREIL
"""

import pandas as pd
import os
from skimage import io, color
from skimage.filters import meijering, sato, frangi, hessian
import matplotlib.pyplot as plt

# Function to find a file by name in a directory and its subdirectories
def find_file(start_dir, filename):
    for dirpath, dirnames, filenames in os.walk(start_dir):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None  # File not found


def original(image, **kwargs):
    """Return the original image, ignoring any kwargs."""
    return image

folder_path = "C:/Users\Hp\Downloads\Images_IMAJ"

# Load the DataFrame from the CSV file
df = pd.read_csv("../Images/IMAJ_image_database.csv")

random_samples = df.sample(n=1)

for sample in random_samples.iloc():
    print(sample)
    
    image_name = sample['Filename']
    image_path = find_file(folder_path, image_name)

    # Load the image
    print(image_path)
    image = io.imread(image_path)
    
    # Convert the image to grayscale
    image_gray = color.rgb2gray(image)
    cmap = plt.cm.gray
    
    plt.rcParams["axes.titlesize"] = "medium"
    axes = plt.figure(figsize=(10, 4)).subplots(2, 9)
    for i, black_ridges in enumerate([True, False]):
        for j, (func, sigmas) in enumerate([
                (original, None),
                (meijering, [1]),
                (meijering, range(1, 5)),
                (sato, [1]),
                (sato, range(1, 5)),
                (frangi, [1]),
                (frangi, range(1, 5)),
                (hessian, [1]),
                (hessian, range(1, 5)),
        ]):
            result = func(image, black_ridges=black_ridges, sigmas=sigmas)
            axes[i, j].imshow(result, cmap=cmap)
            if i == 0:
                title = func.__name__
                if sigmas:
                    title += f"\n\N{GREEK SMALL LETTER SIGMA} = {list(sigmas)}"
                axes[i, j].set_title(title)
            if j == 0:
                axes[i, j].set_ylabel(f'{black_ridges = }')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    
    plt.tight_layout()
    plt.show()