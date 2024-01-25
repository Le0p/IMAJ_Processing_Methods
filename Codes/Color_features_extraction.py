# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:55:36 2023

@author: Leonard TREIL
"""

import os
import sys
import csv

import Color_library
import Image_utils


if __name__ == "__main__":
    
    folder_path = sys.argv[1]
    output_csv = sys.argv[2]
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            
            image_path = os.path.join(folder_path, filename)
       
            # Load image
            image = Image_utils.load_image(image_path)[::4, ::4]
            
            # Compute features
            image_features = []
            
            # Colorimetric features
            color_features = Color_library.extract_KMeans_colors_features(image)
            image_features.extend(color_features)
        
            # open the file in the write mode
            
            with open(output_csv, 'a') as f:
                # create the csv writer
                writer = csv.writer(f)
            
                # write a row to the csv file
                writer.writerow([image_path] + image_features)
        
    

