# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:26:21 2023

@author: Léoanrd TREIL
"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from Dataset_Creation import create_dataset
import Classification_utils


import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")
warnings.filterwarnings("ignore", "The palette list has more values", UserWarning, module="seaborn.axisgrid")


folder_path = "C:/Users\Hp\Downloads\Images_IMAJ"

# Load the DataFrame from the CSV file
df = pd.read_csv("../Images/IMAJ_image_database_with_wu_color_features.csv")

train_list, val_list = create_dataset(
    df,
    year_range=(2017, 2021),
    age_class_range=(0, 5),
    regions=None,
    n_samples=2000,  # Set the number of samples you want
    training_ratio = 0.8, # Set the training/validation ratio
    balanced=True,  # Balance classes and subclasses
    shuffle=True
)

print("\n\n############### Generate datas ###############")

# Generate training and validations datas
X_train, y_year_train, y_age_train, y_region_train = Classification_utils.generate_features_and_labels(df, train_list, "Wu_color_features", folder_path)

X_val, y_year_val, y_age_val, y_region_val = Classification_utils.generate_features_and_labels(df, val_list, "Wu_color_features", folder_path)

le_region = LabelEncoder()
le_year = LabelEncoder()
y_region_train_numeric = le_region.fit_transform(y_region_train)
y_region_val_numeric = le_region.transform(y_region_val)
y_year_train_numeric = le_year.fit_transform(y_year_train)
y_year_val_numeric = le_year.transform(y_year_val)

#Classification_utils.visualize_data(X_train, y_year_train_numeric, vis_type=["UMAP"])

print("\n")

print("\n\n############### Classification ###############")
# Create classifiers for each classification task
year_classifier = RandomForestClassifier()
age_classifier = RandomForestClassifier()
region_classifier = RandomForestClassifier()

# Train the classifiers
year_classifier.fit(X_train, y_year_train_numeric)
age_classifier.fit(X_train, y_age_train)
region_classifier.fit(X_train, y_region_train_numeric)

# Predictions
year_predictions = year_classifier.predict(X_val)
age_predictions = age_classifier.predict(X_val)
region_predictions = region_classifier.predict(X_val)

print("\n")

print("\n\n############### Metrics for age classifier ###############")
Classification_utils.compute_metrics(y_age_val, age_predictions)
print("\n")

print("\n\n############### Metrics for year classifier ###############")
Classification_utils.compute_metrics(y_year_val_numeric, year_predictions, le_year)
print("\n")

print("\n\n############### Metrics for region classifier ###############")
Classification_utils.compute_metrics(y_region_val_numeric, region_predictions, le_region)
print("\n")