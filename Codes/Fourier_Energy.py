# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 08:55:35 2023

@author: Léonard TREIL
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.transform import resize
from skimage.draw import disk
from scipy.optimize import curve_fit

def custom_equation(x, a, alpha):
    return a * (x ** alpha)


# Function to find a file by name in a directory and its subdirectories
def find_file(start_dir, filename):
    for dirpath, dirnames, filenames in os.walk(start_dir):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None  # File not found

folder_path = "C:/Users\Hp\Downloads\Images_IMAJ"
#folder_path = "C:/Users/Hp/Nextcloud/EPFL/Master 3/IMAJ_Processing_Methods/Images"

# Load the DataFrame from the CSV file
df = pd.read_csv("../Images/IMAJ_image_database.csv")

random_samples = df.sample(n=2)

#random_samples = ["IMG_nat1.jpg", "IMG_nat2.jpg", "IMG_nat3.jpg"]

for sample in random_samples.iloc():
    print(sample)
        
    image_name = sample['Filename']
    #image_name = sample
    image_path = find_file(folder_path, image_name)

    # Load the image
    image = io.imread(image_path)
    
    # Convert the image to grayscale
    image_gray = color.rgb2gray(image)
    
    # Resize the image to a manageable size (optional, adjust as needed)
    #image_resized = resize(image_gray, (256, 256))
    
    # Determine the minimum dimension of the image
    min_dimension = min(image_gray.shape)
    
    # Calculate the starting and ending indices for the central square region
    start_idx_1 = (image_gray.shape[0] - min_dimension) // 2
    end_idx_1 = start_idx_1 + min_dimension
    start_idx_2 = (image_gray.shape[1] - min_dimension) // 2
    end_idx_2 = start_idx_2 + min_dimension
    
    # Crop the central square part of the image
    image_gray_square = image_gray[start_idx_1:end_idx_1, start_idx_2:end_idx_2]
    
    # Define a window function (e.g., Hamming or Hann)
    window = np.hamming(min_dimension)
    window_2d = np.sqrt(np.outer(window, window))
    
    # Apply the window to the image 
    image_windowed = image_gray_square * window_2d
    image_windowed -= np.mean(image_windowed)
    
    
    # Compute the Fourier transform
    f_transform = np.fft.fft2(image_windowed)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # Calculate energies based on the surface area of circles
    energies = []
    areas = []
    radii = []
    omega = []
    max_area = (min_dimension // 2) ** 2 * np.pi
    
    area = 1
    while area < max_area:
    #for area in range(1, int(max_area) + 1, 10000):
        # Calculate the corresponding radius for the given area
        area = 1.5 * area
        if area > max_area:
            area = max_area
        
        radius = int(np.sqrt(area / np.pi))

        if radius <= 2:
            continue
        
        # Create a mask for the circle
        mask = np.zeros_like(image_gray_square)
        center = (image_gray_square.shape[0] // 2, image_gray_square.shape[1] // 2)
        rr, cc = disk((center[0], center[1]), int(radius))
        mask[rr, cc] = 1
        mask_sum = np.sum(mask)
        
        # Compute energy by summing the magnitude spectrum within the circle
        if mask_sum != 0:
            energy = np.sum(magnitude_spectrum * mask)/mask_sum
        else:
            continue
            
        energies.append(energy)
        areas.append(area)
        radii.append(radius)
        omega.append(radius*np.pi*2/min_dimension)
    
    energy_derivative = np.gradient(energies)
    
    # Perform linear regression on the log of energy against the log of omega
    coefficients = np.polyfit(np.log10(omega), np.log10(energies), 1)
    poly = np.poly1d(coefficients)
    fit_line = poly(np.log10(omega))
    
    # Perform regression on the derivative against the omega
    params, covariance = curve_fit(custom_equation, omega, energy_derivative)
    optimal_a, optimal_alpha = params
    
    fit_derivative = custom_equation(omega, optimal_a, optimal_alpha)
    
    # Plotting radius against energy
    fig, axs = plt.subplots(1,3, figsize=(16, 6))
    axs[0].imshow(image_windowed)
    axs[0].axis('off')
    axs[1].scatter(np.log10(omega), np.log10(energies), label="Measure")
    axs[1].plot(np.log10(omega), fit_line, color='red', label=f'Fitted Line = {coefficients[0]}')
    axs[1].set_xlabel('Log Omega')
    axs[1].set_ylabel('Log Energy')
    axs[1].set_title('Log Omega vs Log Energy within Circle')
    axs[1].grid(True)
    axs[1].legend()
    axs[2].scatter(omega, energy_derivative, label="Measure")
    axs[2].plot(omega, fit_derivative, color='red', label=f'Fitted Curve = {optimal_alpha}')
    axs[2].set_xlabel('Omega')
    axs[2].set_ylabel('Energy derivative')
    axs[2].set_title('Derivative of Energy with respect to Omega')
    axs[2].grid(True)
    axs[2].legend()
    plt.show()
    
    plt.figure()
    plt.scatter(np.log10(omega), np.log10(energies), label="Measure")
    plt.plot(np.log10(omega), fit_line, color='red', label=f'Fitted Line = {coefficients[0]}')
    plt.xlabel('Log Omega')
    plt.ylabel('Log Energy')
    plt.title('Log Omega vs Log Energy within Circle')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    plt.figure()
    plt.imshow(image_windowed, cmap='gray')
    plt.axis('off')
    plt.show()
    
    plt.figure()
    plt.imshow((magnitude_spectrum - np.min(magnitude_spectrum))/(np.max(magnitude_spectrum) - np.min(magnitude_spectrum)), cmap='gray')
    plt.axis('off')
    plt.show()
