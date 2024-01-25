# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:40:57 2023

@author: Léonard TREIL
"""

import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error

MAXCOLOR = 256
RED = 2
GREEN = 1
BLUE = 0

def create_box(r0, r1, g0, g1, b0, b1):
    return {
        'r0': r0,
        'r1': r1,
        'g0': g0,
        'g1': g1,
        'b0': b0,
        'b1': b1,
        'vol': (r1 - r0) * (g1 - g0) * (b1 - b0)
    }


def hist3d(Ir, Ig, Ib, size):
    vwt = np.zeros((33, 33, 33))
    vmr = np.zeros((33, 33, 33))
    vmg = np.zeros((33, 33, 33))
    vmb = np.zeros((33, 33, 33))
    m2 = np.zeros((33, 33, 33))
    table = [i**2 for i in range(256)]
    Qadd = np.zeros(size, dtype=np.uint16)

    for i in range(size):
        r, g, b = Ir[i], Ig[i], Ib[i]
        inr, ing, inb = (r >> 3) + 1, (g >> 3) + 1, (b >> 3) + 1
        ind = (inr << 10) + (inr << 6) + inr + (ing << 5) + ing + inb
        Qadd[i] = ind
        vwt[inr][ing][inb] += 1
        vmr[inr][ing][inb] += r
        vmg[inr][ing][inb] += g
        vmb[inr][ing][inb] += b
        m2[inr][ing][inb] += table[r] + table[g] + table[b]

    return vwt, vmr, vmg, vmb, m2, Qadd

def m3d(vwt, vmr, vmg, vmb, m2):
    for r in range(1, 33):
        area = np.zeros(33)
        area_r = np.zeros(33)
        area_g = np.zeros(33)
        area_b = np.zeros(33)
        area2 = np.zeros(33)

        for g in range(1, 33):
            line = line_r = line_g = line_b = 0
            line2 = 0.0

            for b in range(1, 33):
                #ind1 = (r << 10) + (r << 6) + r + (g << 5) + g + b

                line += vwt[r][g][b]
                line_r += vmr[r][g][b]
                line_g += vmg[r][g][b]
                line_b += vmb[r][g][b]
                line2 += m2[r][g][b]

                area[b] += line
                area_r[b] += line_r
                area_g[b] += line_g
                area_b[b] += line_b
                area2[b] += line2

                #ind2 = ind1 - 1089  # [r-1][g][b]

                vwt[r][g][b] = vwt[r-1][g][b] + area[b]
                vmr[r][g][b] = vmr[r-1][g][b] + area_r[b]
                vmg[r][g][b] = vmg[r-1][g][b] + area_g[b]
                vmb[r][g][b] = vmb[r-1][g][b] + area_b[b]
                m2[r][g][b] = m2[r-1][g][b] + area2[b]
                
    return vwt, vmr, vmg, vmb, m2

def vol(cube, mmt):
    return (mmt[cube['r1'], cube['g1'], cube['b1']]
            - mmt[cube['r1'], cube['g1'], cube['b0']]
            - mmt[cube['r1'], cube['g0'], cube['b1']]
            + mmt[cube['r1'], cube['g0'], cube['b0']]
            - mmt[cube['r0'], cube['g1'], cube['b1']]
            + mmt[cube['r0'], cube['g1'], cube['b0']]
            + mmt[cube['r0'], cube['g0'], cube['b1']]
            - mmt[cube['r0'], cube['g0'], cube['b0']])


def bottom(cube, dir, mmt):
    if dir == RED:
        return (-mmt[cube['r0'], cube['g1'], cube['b1']]
                + mmt[cube['r0'], cube['g1'], cube['b0']]
                + mmt[cube['r0'], cube['g0'], cube['b1']]
                - mmt[cube['r0'], cube['g0'], cube['b0']])
    elif dir == GREEN:
        return (-mmt[cube['r1'], cube['g0'], cube['b1']]
                + mmt[cube['r1'], cube['g0'], cube['b0']]
                + mmt[cube['r0'], cube['g0'], cube['b1']]
                - mmt[cube['r0'], cube['g0'], cube['b0']])
    elif dir == BLUE:
        return (-mmt[cube['r1'], cube['g1'], cube['b0']]
                + mmt[cube['r1'], cube['g0'], cube['b0']]
                + mmt[cube['r0'], cube['g1'], cube['b0']]
                - mmt[cube['r0'], cube['g0'], cube['b0']])


def top(cube, dir, pos, mmt):
    if dir == RED:
        return (mmt[pos, cube['g1'], cube['b1']]
                - mmt[pos, cube['g1'], cube['b0']]
                - mmt[pos, cube['g0'], cube['b1']]
                + mmt[pos, cube['g0'], cube['b0']])
    elif dir == GREEN:
        return (mmt[cube['r1'], pos, cube['b1']]
                - mmt[cube['r1'], pos, cube['b0']]
                - mmt[cube['r0'], pos, cube['b1']]
                + mmt[cube['r0'], pos, cube['b0']])
    elif dir == BLUE:
        return (mmt[cube['r1'], cube['g1'], pos]
                - mmt[cube['r1'], cube['g0'], pos]
                - mmt[cube['r0'], cube['g1'], pos]
                + mmt[cube['r0'], cube['g0'], pos])

def var(cube, mr, mg, mb, m2, wt):
    dr = vol(cube, mr)
    dg = vol(cube, mg)
    db = vol(cube, mb)
    xx = vol(cube, m2)
    return xx - (dr**2 + dg**2 + db**2) / vol(cube, wt)


def maximize(cube, dir, first, last, whole_r, whole_g, whole_b, whole_w, mr, mg, mb, wt):
    max_value = 0.0
    cut = -1
    base_r = bottom(cube, dir, mr)
    base_g = bottom(cube, dir, mg)
    base_b = bottom(cube, dir, mb)
    base_w = bottom(cube, dir, wt)

    for i in range(first, last):
        half_r = base_r + top(cube, dir, i, mr)
        half_g = base_g + top(cube, dir, i, mg)
        half_b = base_b + top(cube, dir, i, mb)
        half_w = base_w + top(cube, dir, i, wt)

        if half_w == 0:
            continue  # Avoid division by zero; can't split into an empty box

        temp = ((float(half_r)**2 + float(half_g)**2 + float(half_b)**2) / half_w)

        half_r = whole_r - half_r
        half_g = whole_g - half_g
        half_b = whole_b - half_b
        half_w = whole_w - half_w

        if half_w == 0:
            continue  # Avoid division by zero; can't split into an empty box

        temp += ((float(half_r)**2 + float(half_g)**2 + float(half_b)**2) / half_w)

        if temp > max_value:
            max_value = temp
            cut = i

    return max_value, cut


def cut(set1, mr, mg, mb, wt):
    whole_r = vol(set1, mr)
    whole_g = vol(set1, mg)
    whole_b = vol(set1, mb)
    whole_w = vol(set1, wt)

    maxr, cutr = maximize(set1, RED, set1['r0'] + 1, set1['r1'], whole_r, whole_g, whole_b, whole_w, mr, mg, mb, wt)
    maxg, cutg = maximize(set1, GREEN, set1['g0'] + 1, set1['g1'], whole_r, whole_g, whole_b, whole_w, mr, mg, mb, wt)
    maxb, cutb = maximize(set1, BLUE, set1['b0'] + 1, set1['b1'], whole_r, whole_g, whole_b, whole_w, mr, mg, mb, wt)
    
    # Create a new box (set2) as a result of the cut
    set2 = dict(set1)  # Create a copy of set1

    # Determine the best direction to split the box
    if maxr >= maxg and maxr >= maxb:
        if cutr < 0: return False, set1, set2  # No cut possible
        dir = RED
        cut = cutr
    elif maxg >= maxr and maxg >= maxb:
        dir = GREEN
        cut = cutg
    else:
        dir = BLUE
        cut = cutb

    if dir == RED:
        set1['r1'] = cut
        set2['r0'] = cut
    elif dir == GREEN:
        set1['g1'] = cut
        set2['g0'] = cut
    else:
        set1['b1'] = cut
        set2['b0'] = cut

    # Recalculate the volume of the boxes after the cut
    set1['vol'] = (set1['r1'] - set1['r0']) * (set1['g1'] - set1['g0']) * (set1['b1'] - set1['b0'])
    set2['vol'] = (set2['r1'] - set2['r0']) * (set2['g1'] - set2['g0']) * (set2['b1'] - set2['b0'])

    return True, set1, set2


def mark(cube, label, tag):
    for r in range(cube['r0'] + 1, cube['r1'] + 1):
        for g in range(cube['g0'] + 1, cube['g1'] + 1):
            for b in range(cube['b0'] + 1, cube['b1'] + 1):
                ind = (r << 10) + (r << 6) + r + (g << 5) + g + b
                tag[ind] = label


def wu_quantizer_algorithm(img, num_colors):
    
    size = img.shape[0] * img.shape[1]

    Ir = img[:, :, 0].flatten()
    Ig = img[:, :, 1].flatten()
    Ib = img[:, :, 2].flatten()

    # Initialize arrays for histogram and moments
    wt = np.zeros((33, 33, 33), dtype=np.int64)
    mr = np.zeros((33, 33, 33), dtype=np.int64)
    mg = np.zeros((33, 33, 33), dtype=np.int64)
    mb = np.zeros((33, 33, 33), dtype=np.int64)
    m2 = np.zeros((33, 33, 33), dtype=np.float64)

    # Build 3D color histogram and compute moments
    wt, mr, mg, mb, m2, Qadd = hist3d(Ir, Ig, Ib, size)
    wt, mr, mg, mb, m2 = m3d(wt, mr, mg, mb, m2)

    # Partition the color space
    cube = [create_box(0, 32, 0, 32, 0, 32)]*MAXCOLOR
    next = 0
    for i in range(1, num_colors):
        if cube[next]['vol'] < 1:
            break

        cut_flag, set1, set2 = cut(cube[next], mr, mg, mb, wt)
        if cut_flag:
            cube[next] = set1
            cube[i] = set2

        # Find the next box to split
        next = 0
        for j in range(1, i + 1):
            if cube[j]['vol'] > cube[next]['vol']:
                next = j

    # Generate lookup tables (LUTs) for each color box
    lut_r = np.zeros(num_colors, dtype=np.uint8)
    lut_g = np.zeros(num_colors, dtype=np.uint8)
    lut_b = np.zeros(num_colors, dtype=np.uint8)
    tag = np.zeros(33*33*33, dtype=np.uint8)

    for k, box in enumerate(cube[:num_colors]):
        mark(box, k, tag)
        weight = vol(box, wt)
        if weight:
            lut_r[k] = vol(box, mr) // weight
            lut_g[k] = vol(box, mg) // weight
            lut_b[k] = vol(box, mb) // weight

    # Map original image pixels to quantized colors using LUTs
    quantized_img_data = np.zeros_like(img)
    for i in range(size):
        inr, ing, inb = (Ir[i] >> 3) + 1, (Ig[i] >> 3) + 1, (Ib[i] >> 3) + 1
        index = tag[(inr << 10) + (inr << 6) + inr + (ing << 5) + ing + inb]
        quantized_img_data[i // img.shape[1], i % img.shape[1], :] = [lut_r[index], lut_g[index], lut_b[index]]

    quantized_image = Image.fromarray(quantized_img_data, 'RGB')
    
    # Combine them into a single lookup table
    lut = np.array(list(zip(lut_r, lut_g, lut_b)))
    
    return np.array(quantized_image), lut

def extract_wu_colors_features(image):
    
    num_colors_max = 256
    
    num_colors_tab = []
    nmse_tab = []
    colors_tab = []
    images_tab = []
    
    step_size = 1  # Starting step size
    early_stopping_threshold = 0.01  # Threshold for early stopping
    nmse_prev = 0
    
    i = 0
    nmse = 0
    
    while nmse < 0.99 and i < num_colors_max:

        if abs(nmse - nmse_prev) < early_stopping_threshold:
            early_stopping_threshold *= 0.1
            step_size = step_size*2
            #print("Increasing step size")
        elif nmse > 0.98 and step_size > 1:
            step_size = step_size//2
            #print("Reducing step size")
            
        i = i + step_size
        
        if i > num_colors_max:
            i = num_colors_max

        quantized_image, color_tab = wu_quantizer_algorithm(image, i)
        nmse = 1 - mean_squared_error(image.flatten(), quantized_image.flatten()) / np.var(image.flatten())
        
        num_colors_tab.append(i)
        nmse_tab.append(nmse)
        colors_tab.append(color_tab)
        images_tab.append(quantized_image)
        nmse_prev = nmse
    
    optimal_index = -1
    
    if step_size > 1 and len(num_colors_tab) > 1:
        if nmse > 0.99:
            i_start = num_colors_tab[-2]
            insert_index = -1
            
            while step_size != 1:
                step_size //= 2
                i = i_start + step_size   
                
                quantized_image, color_tab = wu_quantizer_algorithm(image, i)
                nmse = 1 - mean_squared_error(image.flatten(), quantized_image.flatten()) / np.var(image.flatten())
                
                num_colors_tab.insert(insert_index, i)
                nmse_tab.insert(insert_index, nmse)
                colors_tab.insert(insert_index, color_tab)
                images_tab.insert(insert_index, quantized_image)
                
                if nmse > 0.99:
                    insert_index -= 1
                else:
                    i_start = i
            
            optimal_index = next(x for x, val in enumerate(nmse_tab) if val > 0.99)
            
        else:
            optimal_index = len(num_colors_tab)-1
            flag = True
            while flag:
                if abs(nmse_tab[optimal_index-1]-nmse_tab[optimal_index]) < 0.001:
                    optimal_index -= 1
                else:
                    flag = False
            
            i_start = num_colors_tab[optimal_index - 1]
            step_size = num_colors_tab[optimal_index] - i_start
            insert_index = optimal_index
            
            while step_size != 1:
                step_size //= 2
                i = i_start + step_size   
                
                quantized_image, color_tab = wu_quantizer_algorithm(image, i)
                nmse = 1 - mean_squared_error(image.flatten(), quantized_image.flatten()) / np.var(image.flatten())
                
                diff1 = abs((nmse_tab[optimal_index - 1]-nmse)/(num_colors_tab[optimal_index - 1]-i))
                diff2 = abs((nmse_tab[optimal_index]-nmse)/(num_colors_tab[optimal_index]-i))
                
                num_colors_tab.insert(optimal_index, i)
                nmse_tab.insert(optimal_index, nmse)
                colors_tab.insert(optimal_index, color_tab)
                images_tab.insert(optimal_index, quantized_image)
                
                if diff1 > diff2:
                    i_start = i_start
                else:
                    i_start = i
                    optimal_index += 1
    
    optimal_color_number = num_colors_tab[optimal_index]
    
    colors_features = [optimal_color_number]
    
    color_perc = []
    for color_i in colors_tab[optimal_index]:
        color_perc.append(np.sum(np.sum(images_tab[optimal_index] == color_i, axis=2) != 0)/(image.shape[0] * image.shape[1]))
        
    color_sort = np.argsort(color_perc)
        
    for i in range(3):
        
        if i == 2 and len(num_colors_tab) == 1:
            colors_features.append(0)
            colors_features.append(0)
            colors_features.append(0)
            colors_features.append(0.0)
            break
        
        j = color_sort[-i-1]
        colors_features.append(colors_tab[optimal_index][j][0])
        colors_features.append(colors_tab[optimal_index][j][1])
        colors_features.append(colors_tab[optimal_index][j][2])
        colors_features.append(np.sum(np.sum(images_tab[optimal_index] == colors_tab[optimal_index][j], axis=2) != 0)/(image.shape[0] * image.shape[1]))

    return colors_features
