# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:40:57 2023

@author: Léonard TREIL
"""


import numpy as np
from PIL import Image

MAXCOLOR = 256
RED = 2
GREEN = 1
BLUE = 0

class Box:
    def __init__(self):
        self.r0 = 0
        self.r1 = 32
        self.g0 = 0
        self.g1 = 32
        self.b0 = 0
        self.b1 = 32
        self.vol = 0

def Hist3d(vwt, vmr, vmg, vmb, m2, Ir, Ig, Ib):
    table = [i ** 2 for i in range(256)]
    Qadd = np.zeros((size,3), dtype=np.uint16)
    for i in range(size):
        r, g, b = Ir[i], Ig[i], Ib[i]
        inr = (r >> 3) + 1
        ing = (g >> 3) + 1
        inb = (b >> 3) + 1
        Qadd[i] = [inr, ing, inb]
        vwt[inr][ing][inb] += 1
        vmr[inr][ing][inb] += r
        vmg[inr][ing][inb] += g
        vmb[inr][ing][inb] += b
        m2[inr][ing][inb] += (table[r] + table[g] + table[b])
        
    return vwt, vmr, vmg, vmb, m2, Qadd

def M3d(vwt, vmr, vmg, vmb, m2):
    area = np.zeros(33, dtype=np.int64)
    area_r = np.zeros(33, dtype=np.int64)
    area_g = np.zeros(33, dtype=np.int64)
    area_b = np.zeros(33, dtype=np.int64)
    area2 = np.zeros(33, dtype=np.float32)

    for r in range(1, 33):
        for i in range(33):
            area[i] = area_r[i] = area_g[i] = area_b[i] = 0

        for g in range(1, 33):
            line2 = line = line_r = line_g = line_b = 0

            for b in range(1, 33):
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

                vwt[r][g][b] = vwt[r-1][g][b] + area[b]
                vmr[r][g][b] = vmr[r-1][g][b] + area_r[b]
                vmg[r][g][b] = vmg[r-1][g][b] + area_g[b]
                vmb[r][g][b] = vmb[r-1][g][b] + area_b[b]
                m2[r][g][b] = m2[r-1][g][b] + area2[b]
    return vwt, vmr, vmg, vmb, m2

def Vol(cube, mmt):
    print(cube.r1)
    return (
        mmt[cube.r1][cube.g1][cube.b1]
        - mmt[cube.r1][cube.g1][cube.b0]
        - mmt[cube.r1][cube.g0][cube.b1]
        + mmt[cube.r1][cube.g0][cube.b0]
        - mmt[cube.r0][cube.g1][cube.b1]
        + mmt[cube.r0][cube.g1][cube.b0]
        + mmt[cube.r0][cube.g0][cube.b1]
        - mmt[cube.r0][cube.g0][cube.b0]
    )

def Bottom(cube, dir, mmt):
    if dir == RED:
        return (
            -mmt[cube.r0][cube.g1][cube.b1]
            + mmt[cube.r0][cube.g1][cube.b0]
            + mmt[cube.r0][cube.g0][cube.b1]
            - mmt[cube.r0][cube.g0][cube.b0]
        )
    elif dir == GREEN:
        return (
            -mmt[cube.r1][cube.g0][cube.b1]
            + mmt[cube.r1][cube.g0][cube.b0]
            + mmt[cube.r0][cube.g0][cube.b1]
            - mmt[cube.r0][cube.g0][cube.b0]
        )
    elif dir == BLUE:
        return (
            -mmt[cube.r1][cube.g1][cube.b0]
            + mmt[cube.r1][cube.g0][cube.b0]
            + mmt[cube.r0][cube.g1][cube.b0]
            - mmt[cube.r0][cube.g0][cube.b0]
        )

def Top(cube, dir, pos, mmt):
    if dir == RED:
        return (
            mmt[pos][cube.g1][cube.b1]
            - mmt[pos][cube.g1][cube.b0]
            - mmt[pos][cube.g0][cube.b1]
            + mmt[pos][cube.g0][cube.b0]
        )
    elif dir == GREEN:
        return (
            mmt[cube.r1][pos][cube.b1]
            - mmt[cube.r1][pos][cube.b0]
            - mmt[cube.r0][pos][cube.b1]
            + mmt[cube.r0][pos][cube.b0]
        )
    elif dir == BLUE:
        return (
            mmt[cube.r1][cube.g1][pos]
            - mmt[cube.r1][cube.g0][pos]
            - mmt[cube.r0][cube.g1][pos]
            + mmt[cube.r0][cube.g0][pos]
        )

def Var(cube):
    print(cube)
    print(mr)
    dr = Vol(cube, mr)
    dg = Vol(cube, mg)
    db = Vol(cube, mb)
    xx = (
        m2[cube.r1][cube.g1][cube.b1]
        - m2[cube.r1][cube.g1][cube.b0]
        - m2[cube.r1][cube.g0][cube.b1]
        + m2[cube.r1][cube.g0][cube.b0]
        - m2[cube.r0][cube.g1][cube.b1]
        + m2[cube.r0][cube.g1][cube.b0]
        + m2[cube.r0][cube.g0][cube.b1]
        - m2[cube.r0][cube.g0][cube.b0]
    )

    return xx - (dr * dr + dg * dg + db * db) / Vol(cube, wt)

def Maximize(cube, dir, first, last, cut, whole_r, whole_g, whole_b, whole_w):
    base_r = Bottom(cube, dir, mr)
    base_g = Bottom(cube, dir, mg)
    base_b = Bottom(cube, dir, mb)
    base_w = Bottom(cube, dir, wt)
    max_val = 0.0
    cut_val = -1

    for i in range(first, last):
        half_r = base_r + Top(cube, dir, i, mr)
        half_g = base_g + Top(cube, dir, i, mg)
        half_b = base_b + Top(cube, dir, i, mb)
        half_w = base_w + Top(cube, dir, i, wt)

        if half_w == 0:
            continue
        else:
            temp = (
                (half_r * half_r + half_g * half_g + half_b * half_b)
                / half_w
            )

        half_r = whole_r - half_r
        half_g = whole_g - half_g
        half_b = whole_b - half_b
        half_w = whole_w - half_w

        if half_w == 0:
            continue
        else:
            temp += (
                (half_r * half_r + half_g * half_g + half_b * half_b)
                / half_w
            )

        if temp > max_val:
            max_val = temp
            cut_val = i

    cut[0] = cut_val
    return max_val

def Cut(set1, set2):
    whole_r = Vol(set1, mr)
    whole_g = Vol(set1, mg)
    whole_b = Vol(set1, mb)
    whole_w = Vol(set1, wt)

    maxr = Maximize(set1, RED, set1.r0 + 1, set1.r1, [0], whole_r, whole_g, whole_b, whole_w)
    maxg = Maximize(set1, GREEN, set1.g0 + 1, set1.g1, [0], whole_r, whole_g, whole_b, whole_w)
    maxb = Maximize(set1, BLUE, set1.b0 + 1, set1.b1, [0], whole_r, whole_g, whole_b, whole_w)

    if maxr >= maxg and maxr >= maxb:
        dir = RED
        if maxr < 0:
            return 0
    elif maxg >= maxr and maxg >= maxb:
        dir = GREEN
    else:
        dir = BLUE

    set2.r1 = set1.r1
    set2.g1 = set1.g1
    set2.b1 = set1.b1

    if dir == RED:
        set2.r0 = set1.r1 = maxr
        set2.g0 = set1.g0
        set2.b0 = set1.b0
    elif dir == GREEN:
        set2.g0 = set1.g1 = maxg
        set2.r0 = set1.r0
        set2.b0 = set1.b0
    else:
        set2.b0 = set1.b1 = maxb
        set2.r0 = set1.r0
        set2.g0 = set1.g0

    set1.vol = (set1.r1 - set1.r0) * (set1.g1 - set1.g0) * (set1.b1 - set1.b0)
    set2.vol = (set2.r1 - set2.r0) * (set2.g1 - set2.g0) * (set2.b1 - set2.b0)
    return 1

def Mark(cube, label, tag):
    for r in range(cube.r0 + 1, cube.r1):
        for g in range(cube.g0 + 1, cube.g1):
            for b in range(cube.b0 + 1, cube.b1):
                tag[(r << 10) + (r << 6) + r + (g << 5) + g + b] = label












# Parameters
image_path = 'img_Test.png'
num_colors_max = 10

# Load the image
image = np.array(Image.open(image_path))

# Input R, G, B components into Ir, Ig, Ib (assuming you have those arrays)
Ir = image[:,:, RED].reshape(-1)
Ig = image[:,:, GREEN].reshape(-1)
Ib = image[:,:, BLUE].reshape(-1)
size = len(Ir)


K = int(input("no. of colors: "))
wt = np.zeros((33, 33, 33), dtype=np.int64)
mr = np.zeros((33, 33, 33), dtype=np.int64)
mg = np.zeros((33, 33, 33), dtype=np.int64)
mb = np.zeros((33, 33, 33), dtype=np.int64)
m2 = np.zeros((33, 33, 33), dtype=np.float32)

wt, mr, mg, mb, m2, Qadd = Hist3d(wt, mr, mg, mb, m2, Ir, Ig, Ib)
print("Histogram done")

Ig, Ib, Ir = None, None, None  # Ir, Ig, Ib should be arrays with your image data

wt, mr, mg, mb, m2 = M3d(wt, mr, mg, mb, m2)
print("Moments done");

cube = [Box() for _ in range(MAXCOLOR)]
lut_r = np.zeros(MAXCOLOR, dtype=np.uint8)
lut_g = np.zeros(MAXCOLOR, dtype=np.uint8)
lut_b = np.zeros(MAXCOLOR, dtype=np.uint8)
tag = np.zeros(33 * 33 * 33, dtype=np.uint8)
vv = np.zeros(MAXCOLOR, dtype=np.float32)

next = 0
for i in range(1,K):
    print(cube[next].r1)
    if Cut(cube[next], cube[i]):
        if(cube[next].vol>1):
            print(cube[next].r1)
            vv[next] = Var(cube[next])
        else:
            vv[next] = 0.0
        
        if(cube[i].vol>1):
            vv[i] = Var(cube[i])
        else:
            vv[i] = 0.0
    else:
        vv[next]=0.0
        i-=1
    
    next = 0
    temp = vv[0]
    
    for k in range(1,K):
        if vv[k] > temp:
            temp = vv[k]
            next = k
    
    if temp <= 0.0:
        K=i+1
        print(f"Only got {K} boxes\n")
        break
    
print("Partition Done")   
    
#%%   
for k in range(1,K):
    Mark(cube[k], k, tag)
    weight = Vol(cube[k], wt)
    if weight:
        lut_r[k] = Vol(cube[k], mr) // weight
        lut_g[k] = Vol(cube[k], mg) // weight
        lut_b[k] = Vol(cube[k], mb) // weight
    else:
        print(f"bogus box {k}")
        lut_r[k] = lut_g[k] = lut_b[k] = 0

#%%
for i in range(size):
    Qadd[i] = tag[Qadd[i]]