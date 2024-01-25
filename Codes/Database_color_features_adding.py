# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 06:16:22 2023

@author: Leonard TREIL
"""

import pandas as pd
import os
from tqdm import tqdm

def extract_csv_list(folder_path):
    
    csv_list = []
    
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.csv'):
                csv_list.append(os.path.join(dirpath, filename))
    
    sorted(csv_list)
    
    return csv_list


folder_path = "C:/Users\Hp\Downloads\Images_IMAJ\csv_files"

# Load the DataFrame from the CSV file
df = pd.read_csv("../Images/IMAJ_image_database_with_color_features.csv")

csv_list = extract_csv_list(folder_path)
count = 0

for csv_path in tqdm(csv_list):
    df1 = pd.read_csv(csv_path, header=None)
    for sample in df1.iloc():
        img_name = sample[0].split("/")[-1]
        color_features = list(sample[1:len(sample)])

        desired_index = df.index[df['Filename'] == img_name].tolist()
        
        if desired_index:
            df.loc[desired_index, 'Wu Color Features'] = pd.Series([color_features], index=df.index[desired_index])
        else:
            print(f"Filename {img_name} not found in the DataFrame")
            count+=1
            print(count)

df.to_csv("../Images/IMAJ_image_database_with_wu_color_features.csv", index=False)