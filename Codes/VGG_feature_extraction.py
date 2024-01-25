# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 07:19:43 2023

@author: Léonard TREIL
"""

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

model = VGG16(weights='imagenet', include_top=False)

img_path = "../Images/IMG_Test.png"
img = image.load_img(img_path, target_size=(224,224))
plt.imshow(img)
plt.show()
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
print(features.shape)
print(features)