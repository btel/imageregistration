#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

from warps_py import warp_int 

from skimage import transform
from skimage import io
import os
img_path = '/Users/bartosz/Desktop/TREE/registration/'

if __name__ == '__main__':

    fname = 'transforms.txt'

    transforms = np.recfromcsv(fname, names=None)

    imgs = {}

    for t in transforms:
        data = list(t)
        date, target, ref = data[:3]

        imgs[target] = data[3:]


    for img, params in imgs.items():
        im = io.imread(os.path.join(img_path,img))
        matrix = np.array(params).reshape(3,3)
        trans = transform.AffineTransform(matrix=matrix)
        img_reshaped = warp_int(im, trans.inverse)
        io.imsave(img, img_reshaped)

