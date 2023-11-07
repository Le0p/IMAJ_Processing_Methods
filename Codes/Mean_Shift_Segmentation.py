# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:12:15 2023

@author: Léonard TREIL
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import MeanShift, estimate_bandwidth
import time


def scatterplot_matrix(data, names=None):
    
    num_dim = np.shape(data)[1]
    
    fig, axs = plt.subplots(num_dim, num_dim)
    
    for i in range(num_dim):
        for j in range(num_dim):
            axs[i,j].plot(data[:,i], data[:,j], marker='+', linestyle='')
            
    plt.show()
    

file_path = "../Images/"
        
imagename = 'IMG_test'

# Load the image
image = np.array(Image.open(file_path + imagename + ".png"))

# Making sure that there is no "transparency" channel
if image.shape[-1] == 4:
    image = image[..., :3]
    
# Reshape the image to a 2D array of pixels
pixels = image.reshape(-1, 3)
#scatterplot_matrix(pixels, ['R', 'G', 'B'])

# Apply Mean-Shift
bandwidth = estimate_bandwidth(pixels, quantile=0.2, n_samples=500)

start_time = time.time()
ms = MeanShift(bandwidth=bandwidth)
ms.fit(pixels[:20000])

print("--- %s seconds ---" % (time.time() - start_time))

cluster_centers = ms.cluster_centers_

quantized_image = np.reshape(cluster_centers[ms.predict(pixels)], image.shape)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(quantized_image.astype(np.uint8))
plt.title('Quantized Image')
plt.axis('off')

plt.show()