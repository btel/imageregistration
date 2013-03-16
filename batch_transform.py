#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

from warps_py import warp_int 

from skimage import transform
from skimage import io
import os
import sys

if __name__ == '__main__':

    fname = 'transforms.txt'

    _, path_src, path_target = sys.argv

    transforms = np.recfromcsv(fname, names=None)

    imgs = []

    for t in transforms:
        data = list(t)
        date, target, ref = data[:3]

        imgs.append((target, ref, data[3:]))


    for img, ref, params in imgs:
        im = io.imread(os.path.join(path_src,img))
        matrix = np.array(params).reshape(3,3)
        trans = transform.AffineTransform(matrix=matrix)
        img_reshaped = warp_int(im, trans.inverse)
        
        save_dir = os.path.join(path_target, ref)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_path = os.path.join(save_dir, img)
        i = 1
        
        while os.path.exists(file_path):
            root, ext = os.path.splitext(img)
            img_name = "".join([root , "_reg%d" % i, ext])

            file_path = os.path.join(save_dir, img_name)
            i+=1

        io.imsave(file_path, img_reshaped)
        print "Processed image %s and saved to %s" % (img, file_path)

