# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 08:39:12 2023

@author: Léonard TREIL
"""

import numpy as np
from PIL import Image, ImageDraw

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.2
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1
        
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

# Image dimensions
width, height = 800, 600
nb_images = 10

num_shapes_range = [3, 8]
center_range = [[100, 100], [width-100, height-100]]
radius_range = [50, 100]
color_choice = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 0), (255, 255, 255)]

filename = "images_info.txt"
file = open(filename, 'w')
file.write("Name; number of shapes; number of colors; \n")

#%%
# Create circle images
for n in range(nb_images):
    # Create a blank image with a white background
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    
    num_shapes = np.random.randint(num_shapes_range[0], num_shapes_range[1])   
    
    circles_center = []
    circles_radius = []
    circles_color = []
    circles_color_num = []
    for i in range(num_shapes):
        circles_radius.append(np.random.randint(radius_range[0], radius_range[1]))
        
        center_flag = True
        while(center_flag):
            new_center = (np.random.randint(center_range[0][0], center_range[1][0]),np.random.randint(center_range[0][1], center_range[1][1]))
            center_flag = False
            for center in circles_center:
                if new_center[0] < center[0]+100 and new_center[0] > center[0]-100 and new_center[1] < center[1]+100 and new_center[1] > center[1]-100:
                    center_flag = True
                    break  
        circles_center.append(new_center)
        circles_color_num.append(np.random.randint(0, len(color_choice)-1))
        circles_color.append(color_choice[circles_color_num[i]])
    
    # Draw circles with different colors and some with noise
    for i in range(num_shapes):
        x = circles_center[i][0]
        y = circles_center[i][1]
        radius = circles_radius[i] 
        color = circles_color[i]
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=color,
            outline=None,
        )
    
    np_img = np.array(image)
    
    # for i in range(3, num_circles):
    #     x = circles_center[i][0]
    #     y = circles_center[i][1]
    #     radius = circles_radius[i]
        
    #     np_img[(y - radius):(y + radius), (x - radius):(x + radius)] = noisy("gauss", np_img[(y - radius):(y + radius), (x - radius):(x + radius)])
    
    image = Image.fromarray(np_img)
    # Save the image
    image.save(f"Test_images_{n}.png")
    
    # Display the image
#    image.show()

    num_color = len(np.unique(circles_color_num))
    file.write(f"Test_images_{n}; {num_shapes}; {num_color}; \n")


# Create square images
for n in range(nb_images, nb_images*2):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    
    num_shapes = np.random.randint(num_shapes_range[0], num_shapes_range[1])   
    
    circles_center = []
    circles_radius = []
    circles_color = []
    circles_color_num = []
    for i in range(num_shapes):
        circles_radius.append(np.random.randint(radius_range[0], radius_range[1]))
        
        center_flag = True
        while(center_flag):
            new_center = (np.random.randint(center_range[0][0], center_range[1][0]),np.random.randint(center_range[0][1], center_range[1][1]))
            center_flag = False
            for center in circles_center:
                if new_center[0] < center[0]+100 and new_center[0] > center[0]-100 and new_center[1] < center[1]+100 and new_center[1] > center[1]-100:
                    center_flag = True
                    break  
        circles_center.append(new_center)
        circles_color_num.append(np.random.randint(0, len(color_choice)-1))
        circles_color.append(color_choice[circles_color_num[i]])
    
    for i in range(num_shapes):
        x = circles_center[i][0]
        y = circles_center[i][1]
        radius = circles_radius[i] 
        color = circles_color[i]
        draw.rectangle(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=color,
            outline=None,
        )
    
    np_img = np.array(image)
    image = Image.fromarray(np_img)
    image.save(f"Test_images_{n}.png")
    
    num_color = len(np.unique(circles_color_num))
    file.write(f"Test_images_{n}; {num_shapes}; {num_color}; \n")

# Create different size band images
for n in range(nb_images*2, nb_images*3):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)   
    
    circles_center = []
    circles_radius = []
    circles_color = []
    circles_color_num = []
    
    x = 0
    num_shapes = 0
    not_fill_flag = True
    while not_fill_flag:
        num_shapes += 1
        radius = np.random.randint(radius_range[0], radius_range[1])
        color_choice_num = np.random.randint(0, len(color_choice))
        color = color_choice[color_choice_num]
        circles_radius.append(radius)
        circles_color.append(color)
        circles_color_num.append(color_choice_num)
        
        if x + radius > width:
            not_fill_flag = False
            radius = width-x
            
        draw.rectangle(
            [(x, 0), (x + radius, height)],
            fill=color,
            outline=None,
        )
        
        x += radius
        
    
    np_img = np.array(image)
    image = Image.fromarray(np_img)
    image.save(f"Test_images_{n}.png")
    
    num_color = len(np.unique(circles_color_num))
    file.write(f"Test_images_{n}; {num_shapes}; {num_color}; \n")

#%%
# Create same size band images
for n in range(nb_images*3, nb_images*4):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)   
    
    circles_color = []
    circles_color_num = []
    
    num_shapes = np.random.randint(5, 15)
    radius = width/num_shapes
    for i in range(num_shapes):
        
        x = int(round(i*radius))
        
        color_choice_flag = True
        while color_choice_flag:
            color_choice_num = np.random.randint(0, len(color_choice))
            if i == 0 or circles_color_num[i-1] != color_choice_num:
                color_choice_flag = False
                
        color = color_choice[color_choice_num]
        circles_color.append(color)
        circles_color_num.append(color_choice_num)
            
        draw.rectangle(
            [(x, 0), (int(round(x + radius)), height)],
            fill=color,
            outline=None,
        )
        
    
    np_img = np.array(image)
    image = Image.fromarray(np_img)
    image.save(f"Test_images_{n}.png")
    
    num_color = len(np.unique(circles_color_num))
    file.write(f"Test_images_{n}; {num_shapes}; {num_color}; \n")   


file.close()