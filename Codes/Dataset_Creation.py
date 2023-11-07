# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 08:54:57 2023

@author: Léonard TREIL
"""

import pandas as pd
import numpy as np

def create_dataset(df, year_range, age_class_range, regions=None, n_samples=None, balanced=True):
    """
    Create a dataset from the image data DataFrame with optional filters and balancing.

    Parameters:
    - df: DataFrame containing image data.
    - year_range: Tuple (min_year, max_year) for the range of years to include.
    - age_range: Tuple (min_age, max_age) for the range of ages to include.
    - regions: List of world regions to include (e.g., ["Northern America", "Europe"]).
      If None, all regions are included.
    - n_samples: Number of samples to select. If None, use all available samples.
    - balanced: If True, ensure a balanced distribution of classes and subclasses.

    Returns:
    - train_list: List of file paths for the training dataset.
    - val_list: List of file paths for the validation dataset.
    """

    # Filter by year and age
    df_filtered = df[
        (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
        (df['AgeClass'] >= age_class_range[0]) & (df['AgeClass'] <= age_class_range[1])
    ]

    # Filter by regions
    if regions is not None:
        df_filtered = df_filtered[df_filtered['World Region'].isin(regions)]
    
    # Group by 'Age' and 'Year', and calculate the size (count) of each combination
    class_counts = df_filtered.groupby(['AgeClass', 'Year', 'World Region']).size().reset_index(name='Count')
    print(class_counts)
    print(len(class_counts))

    # Initialize lists for the training and validation sets
    train_list = []
    val_list = []

    # Randomly select samples with balanced classes and subclasses
    if n_samples is not None and n_samples > 0:
        for cls in class_counts.iloc:
            print(cls)
            cls_samples = df_filtered[df_filtered['AgeClass'] == cls['AgeClass']]
            cls_samples = cls_samples[cls_samples['Year'] == cls['Year']]
            cls_samples = cls_samples[cls_samples['World Region'] == cls['World Region']]
            print(cls_samples)

            if balanced:
                n_samples_per_class = min(
                    n_samples // len(class_counts),
                    cls['Count']
                )
            else:
                n_samples_per_class = n_samples // len(class_counts)

            class_samples = cls_samples.sample(n_samples_per_class)
            train_samples = class_samples.sample(frac=0.8)
            val_samples = class_samples.drop(train_samples.index)

            train_list.extend(train_samples['Filename'].tolist())
            val_list.extend(val_samples['Filename'].tolist())

    return train_list, val_list

# Load the DataFrame from the CSV file
df = pd.read_csv("image_data.csv")

train_list, val_list = create_dataset(
    df,
    year_range=(2017, 2021),
    age_class_range=(0, 5),
    regions=None,
    n_samples=1000,  # Set the number of samples you want
    balanced=True  # Balance classes and subclasses
)
