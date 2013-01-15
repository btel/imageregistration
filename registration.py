#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage import transform, data
from scipy import optimize
from skimage.io import Image


def normalize(dx, dy, s, angle):
    return dx/10., dy/10., s*10, angle/np.pi
def denorm(x):
    dx, dy, s, angle =  x[0]*10., x[1]*10.,x[2]/10. , x[3]*np.pi
    return np.array([dx, dy, s, angle])


def get_transform(dx, dy, s, angle, img_size):
    shift_x, shift_y = img_size[0]/2, img_size[1]/2
    trans1 = transform.AffineTransform(translation=(shift_x, shift_y))
    trans2 = transform.AffineTransform(rotation=angle)
    trans3 = transform.AffineTransform(translation=(-shift_x, -shift_y))
    trans4 = transform.AffineTransform(scale=(s, s))
    trans5 = transform.AffineTransform(translation=(dx, dy))
    trans = trans5 + trans3 +trans4+ trans2 + trans1
    return trans


def get_simple_transform(x):
    dx, dy, s, rot = x
    return transform.AffineTransform(rotation=rot,
                                     scale=(s, s),
                                     translation=(dx, dy))

def mean_sq_diff(img1, img2):
    return ((img1-img2)**2)[img1>0].mean()

def corr(img1, img2):
    #img2 = img2[img1>0]
    #img1 = img1[img1>0]
    return -np.corrcoef(img1, img2)[0][1]

def transform_and_compare(img, img_ref, params, obj_func=mean_sq_diff):
    #new_transform = get_transform(*params)
    new_transform = get_simple_transform(params)
    img_new = transform.warp(img, new_transform)
    return mean_sq_diff(img_new, img_ref)


def test():
    img = skimage.img_as_float(data.lena())
    img_size = img.shape[:2]

    trans = get_transform(20,15,1.05, 0.02, img_size)
    img_transformed = transform.warp(img, trans)
    obj_func = lambda x: transform_and_compare(img_transformed, img, x)
    x0 = np.array([0,0,1, 0])
    results = optimize.fmin_bfgs(obj_func, x0)

    transform_estimated = get_simple_transform(results) 
    transform_optimal = transform.AffineTransform(np.linalg.inv(trans._matrix))
    params_optimal = np.concatenate([transform_optimal.translation,
                                    transform_optimal.scale[0:1],
                                    [transform_optimal.rotation]])
    img_registered = transform.warp(img_transformed, 
                                    transform_estimated)
    err_original = mean_sq_diff(img_transformed, img)
    err_optimal = transform_and_compare(img_transformed, img, params_optimal) 
    err_actual = transform_and_compare(img_transformed, img, results) 
    err_relative = err_optimal/err_original
    
    print "Params optimal:", params_optimal
    print "Params estimated:", results
    print "Error without registration:", err_original
    print "Error of optimal registration:", err_optimal 
    print "Error of estimated transformation %f (%.2f %% of intial)" % (err_actual,
                                                            err_relative*100.)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img_transformed)
    plt.subplot(122)
    plt.imshow(img_registered)
if __name__ == '__main__':
    test()
