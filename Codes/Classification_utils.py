# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:07:12 2024

@author: Hp
"""

import umap
import ast
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as image_VGG16
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, jaccard_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import Color_library
import Wu_Color_library
import Image_utils

def extract_VGG_features(image_path, model):
    
    img = image_VGG16.load_img(image_path, target_size=(224,224))
    x = image_VGG16.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    
    return np.reshape(features, (-1))

def generate_features_and_labels(df, sample_list, feature_type, folder_path):
    
    y_year = []
    y_age = []
    y_region = []
    X = []
    
    if feature_type == "VGG_features":
        # Load VGG16 model
        model = VGG16(weights='imagenet', include_top=False)

    for image_name in tqdm(sample_list):
        
        # Load classification categories
        sample = df[df['Filename'] == image_name]
        y_year.append(sample.iloc[0]['Year'])
        y_age.append(sample.iloc[0]['AgeClass'])
        y_region.append(sample.iloc[0]['Balanced Region'])
        
        # Load image
        image_path = Image_utils.find_file(folder_path, image_name)
        
        # Compute features
        image_features = []
        
        # Colorimetric features
        if feature_type == "K-Means_color_features":
            # For pre-extracted features
            color_features = ast.literal_eval(sample.iloc[0]['Color Features'])
            image_features.append(color_features)
            # Can also be extracted live with :
            # image = Image_utils.load_image(image_path)
            # color_features = Color_library.extract_KMeans_colors_features(image)
            
        # Colorimetric features
        if feature_type == "Wu_color_features":
            # For pre-extracted features
            wu_color_features = ast.literal_eval(sample.iloc[0]['Wu Color Features'])
            image_features.append(wu_color_features)
            # Can also be extracted live with :
            # image = Image_utils.load_image(image_path)
            # color_features = Wu_Color_library.extract_wu_colors_features(image)
        
        # VGG features
        if feature_type == "VGG_features":
            VGG_features = extract_VGG_features(image_path, model)
            image_features.extend(VGG_features)
        
        X.append(image_features)
    
    X = np.array(X)

    if len(X.shape)>2:
        X = X.squeeze()

    y_year = np.array(y_year)
    y_age = np.array(y_age)
    y_region = np.array(y_region)
    
    return X, y_year, y_age, y_region

def compute_metrics(y_true, predicted_labels, label_encoder=None):
    
    if label_encoder == None:
        classes = np.unique(y_true)
    else:
        classes = label_encoder.classes_
        
    accuracy = accuracy_score(y_true, predicted_labels)
    print(f"Accuracy: {accuracy:.5f}")

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, predicted_labels, average='weighted')
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1-Score: {f1_score:.5f}")
    
    print("True Positive :", np.sum(y_true == predicted_labels)/len(y_true))
    
    print(classes)
    print("Jaccard Score : ", jaccard_score(y_true, predicted_labels, average=None))


    conf_matrix = confusion_matrix(y_true, predicted_labels)
    plt.figure(figsize=(16, 12))
    sns.heatmap(conf_matrix, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def visualize_data(X, y, vis_type=["UMAP", "TSNE", "PCA"]):
    
    for i in range(len(vis_type)):
        if vis_type[i] == "UMAP":
            umap_emb = umap.UMAP(n_components=4)
            X_umap = umap_emb.fit_transform(X)
            
            df1 = pd.DataFrame(X_umap)
            df1['fit'] = y
            g = sns.pairplot(df1, hue='fit', palette=sns.color_palette("tab10"))
            g.fig.suptitle('UMAP Visualization', y=1.08)
            plt.show()
            
        elif vis_type[i] == "TSNE":
            tsne = TSNE(n_components=3)
            X_tsne = tsne.fit_transform(X)
            
            df1 = pd.DataFrame(X_tsne)
            df1['fit'] = y
            g = sns.pairplot(df1, hue='fit', palette=sns.color_palette("tab10"))
            g.fig.suptitle('TSNE Visualization')
            plt.show()
            
        elif vis_type[i] == "PCA":
            pca = PCA(n_components=4)
            X_pca = pca.fit_transform(X)
            
            df1 = pd.DataFrame(X_pca)
            df1['fit'] = y
            g = sns.pairplot(df1, hue='fit', palette=sns.color_palette("tab10"))
            g.fig.suptitle('PCA Visualization')
            plt.show()
