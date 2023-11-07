# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 07:49:23 2023

@author: Léonard TREIL
"""

import os
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

data = []
image_folder = "E:\Images_IMAJ"

# Load the CSV file with country code information
country_codes_df = pd.read_csv("../world-regions.csv")

# Create a dictionary to map ISO3 codes to country names and world regions
country_info_dict = {}
for index, row in country_codes_df.iterrows():
    iso3 = row["ISO3"]
    country_name = row["NAME"]
    world_region = row["UNSubRegionLabel"]
    country_info_dict[iso3] = {"CountryName": country_name, "WorldRegion": world_region}
    
# Define a mapping between age class strings and numerical values
age_class_mapping = {
    "3-5": 1,
    "6-9": 2,
    "10-13": 3,
    "14-17": 4,
    "18-25": 5
}

# Define a function to apply the mapping and create the new column
def map_age_class(age_class_str):
    return age_class_mapping.get(age_class_str, 6)  # Assign 6 for unknown or unclassified
    
for root, dirs, files in os.walk(image_folder):
    print(root)
    for file in tqdm(files):
        if file.endswith(".jpg"):
            file_path = os.path.join(root, file)
            filename = file
            temp = filename.split("_")
            year = int(temp[0])
            age = temp[1]
            age_class = map_age_class(age)
            country_code = temp[3]
            
            country_info = country_info_dict.get(country_code, {"CountryName": "Unknown", "WorldRegion": "Unknown"})
            country_name = country_info["CountryName"]
            world_region = country_info["WorldRegion"]
            
            image = np.array(Image.open(file_path))
            size = image.shape[:2]
            mean_color = np.mean(image, axis=(0, 1))
            variance = np.var(image, axis=(0, 1))
            
            data.append([filename, year, age, age_class, country_code, country_name, world_region, size, mean_color, variance])

column_names = ["Filename", "Year", "Age", "AgeClass", "Country Code", "Country Name", "World Region", "Size", "Mean Color", "Variance"]
df = pd.DataFrame(data, columns=column_names)

df.to_csv("image_data.csv", index=False)
