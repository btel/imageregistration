#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.widgets import Button
from matplotlib.widgets import RectangleSelector

import skimage
from skimage import img_as_ubyte, img_as_float
from skimage import io
from skimage.util.shape import view_as_blocks
from skimage import transform

from registration import corr

from scipy import optimize
from scipy import ndimage

from numpy.lib.stride_tricks import as_strided
from numpy import nan

import os
from itertools import cycle

from scipy.stats import gaussian_kde
from scipy.integrate import dblquad

from warps_py import warp_int 

import logging

def gen_checkerboard(img1, img2, n_blks):
    img_check = view_as_blocks(img1.copy(), n_blks+(3,))
    img2_blks = view_as_blocks(img2, n_blks+(3,))

    #import pdb; pdb.set_trace()
    img_check[:-1:2,:-1:2, :] = img2_blks[:-1:2,:-1:2,:]
    img_check[1::2,1::2, :] = img2_blks[1::2,1::2,:]
    return as_strided(img_check, img1.shape, img1.strides)

def imshow(im, ax):
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])


def kernel_correlation(img, patch):
    
    patch = patch.copy()
    patch -= patch.mean()
    patch /= patch.std()

    def _kernel(p1=img):
        p1 = p1.copy()
        p1 -= p1.mean(); p1 /= p1.std()
        c = np.mean(p1*patch)
        return c
    return _kernel

def kernel_mutualinformation(img,patch, n=50):
    bins_x = np.linspace(img.min(), img.max(), n)
    bins_y = np.linspace(patch.min(), patch.max(), n)
    eps = np.finfo(np.float64).eps
    dx = (img.max()-img.min())*1./n 
    dy = (patch.max()-patch.min())*1./n 
    
    y = patch.flatten()
    ny, _ = np.histogram(y, bins_y, density=True)
    
    def _kernel(x=img):
        x = x.flatten()
        try:
            nx, _ = np.histogram(x, bins_x, density=True)
            nxy, _,_ = np.histogram2d(x,y,[bins_x, bins_y])
            nxy /= 1.*(np.sum(nxy)*dx*dy)
        except ValueError:
            return 0
        aux = nxy*1./(nx[:,None]*ny[None,:])
        aux[np.isnan(aux)] = 0
        mi = np.sum(nxy*np.log(aux+eps))*dx*dy
        return mi

    return _kernel

def correlate(img, patch,kernel=kernel_correlation):
    r,c = img.shape

    out = np.zeros((r,c), dtype=img.dtype)

    p_r, p_c = patch.shape
   
    K = kernel(img, patch)
    for i in range(p_r/2,r-p_r+p_r/2+1):
        for j in range(p_c/2, c-p_c+p_c/2+1):
            r_l, r_h = i-p_r/2, i+p_r-p_r/2 
            c_l, c_h = j-p_c/2, j+p_c-p_c/2 
            im = img[r_l:r_h, c_l:c_h] 
            out[i,j] = K(im)

    out[np.isnan(out)]=0
    return out


class LandmarkSelector:

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    def __init__(self, ax, im):

        self.xs = []
        self.ys = []
        self.markers = []
        self.img = im

        self.color_iter = cycle(self.colors)

        self.ax = ax
        self.fig = ax.figure
        self.marker_radius = 50

        self.picker_radius = 100
        self.zoom_factor = 5.
        imshow(im, ax)

        self._dragged = None
        self.zoom_axes = None
                 
        self.click_ev = ax.figure.canvas.mpl_connect('button_press_event',
                                       self._onclick)
        self.motion_ev = ax.figure.canvas.mpl_connect('motion_notify_event',
                                       self._onmotion)
        self.release_ev = ax.figure.canvas.mpl_connect('button_release_event',
                                       self._onrelease)

    def _onclick(self, event):
        if event.inaxes==self.ax:
            if event.button==1:
                self._left_click(event)
            elif event.button==2:
                self._right_click(event)

    def _picker(self, xy):
        x, y = xy
        dist = (np.abs(np.array(self.xs)-x)+
                np.abs(np.array(self.ys)-y))
        if (dist<self.picker_radius).any():
            return np.argmin(np.ma.masked_invalid(dist))
        else:
            return


    def _left_click(self, event):
        x,y = event.xdata, event.ydata
        artist_idx = self._picker((x,y))
        if artist_idx is not None:
            self._dragged = self.markers[artist_idx]
        else:
            self._dragged = self.add_landmark(x,y)
        self.zoom_image((x,y))
    
    def _right_click(self, event):
        x,y = event.xdata, event.ydata
        artist_idx = self._picker((x,y))
        
        if artist_idx is not None:
            self.remove_landmark(artist_idx)


    def _onmotion(self, event):
        if self._dragged:
            i = self.markers.index(self._dragged)
            self._dragged = self.update_landmark(i, event.xdata, 
                                                    event.ydata)

    def _onrelease(self, event):
        if self._dragged:
            self._dragged.set_radius(self.marker_radius)
            self._dragged = None
            self.ax.autoscale(True)
            self.marker_radius *= self.zoom_factor
            for m in self.markers:
                m.set_radius(self.marker_radius)
            self.ax.figure.canvas.draw()

    def zoom_image(self, center):
        x,y = center
        zoom = self.zoom_factor
        height, width, _ = self.img.shape
        new_height = height*1./zoom
        new_width = width*1./zoom

        #solution to equation (y-top)/new_height == y*1./height
        top = y*(-1./height*new_height+1)
        left = x*(-1./width*new_width+1)

        self.ax.set_xlim([left, left+new_width])
        self.ax.set_ylim([top+new_height, top])
        self.marker_radius /= 1.*zoom
        for m in self.markers:
            m.set_radius(self.marker_radius)
        self.ax.figure.canvas.draw()

    def remove_landmark(self, i):
        
        #removing just replaces coordinates with nans and
        # removes the artist from axes, but not from the list
        # (we need it to keep its color)
        self.xs[i] = nan
        self.ys[i] = nan
        
        m = self.markers[i]
        m.remove()
        
        self.ax.figure.canvas.draw()

    def remove_all_landmarks(self):
        for m in self.markers:
            try:
                m.remove()
            except ValueError:
                pass
        self.xs = []
        self.ys = []
        self.markers = []
        self.color_iter = cycle(self.colors)
        self.ax.figure.canvas.draw()


    def update_landmark(self, i, x, y):

        self.xs[i] = int(x)
        self.ys[i] = int(y)
       
        m = self.markers[i]
        color = m.get_facecolor()
        
        try:
            m.remove()
        except ValueError:
            #marker was already removed
            pass

        new_marker = self._add_patch((x,y), color)
        self.markers[i] = new_marker
        self.ax.figure.canvas.draw()

        return new_marker

    def _add_patch(self, xy, color):
        patch = patches.Circle(xy, self.marker_radius, 
                               edgecolor='none', 
                              facecolor=color)
        self.ax.add_patch(patch)
        return patch
    
    def add_landmark(self, x,y):
        
        try:
            i = self.xs.index(nan)
            patch = self.update_landmark(i, x,y)
            return patch
        except ValueError:
            pass

        self.xs.append(int(x))
        self.ys.append(int(y))
        
        color = next(self.color_iter)

        patch = self._add_patch((x,y), color)
        self.markers.append(patch)

        self.ax.figure.canvas.draw()

        return patch
    
    def _find_landmark(self, img_patch, xy, sub_sample, r):
        
        h,w = self.img_float.shape
        if xy is not None:
            p_h, p_w = img_patch.shape
            x,y = xy
            b,l = np.maximum([0,0], [y-r-p_h/2,x-r-p_w/2])
            t,r = np.minimum([h,w], [y+r+p_h/2,x+r+p_w/2])
            img_float = self.img_float[b:t, l:r]
        else:
            l, b = 0,0
            img_float = self.img_float
        
        img_float = img_float[::sub_sample,::sub_sample]
        img_patch = img_patch[::sub_sample,::sub_sample]
        
        corr = correlate(img_float, img_patch,
                         kernel=kernel_correlation)

        ax1 = fig_debug.add_subplot(221)
        ax1.imshow(corr)
        ax2 = fig_debug.add_subplot(222)
        ax2.imshow(img_float)
        ax2 = fig_debug.add_subplot(223)
        ax2.imshow(img_patch)
        fig.canvas.draw()
        raw_input('press enter to continue')

        y, x = np.unravel_index(corr.argmax(), corr.shape)
        return int((x+0.5)*sub_sample)+l, int((y+0.5)*sub_sample)+b

    def find_landmark(self, img_patch, xy=None, r=100):
        subsample = 4
        
        if not hasattr(self, 'img_float'):
            self.img_float = img_as_float(self.img[:,:,:].mean(2))

        x,y = self._find_landmark(img_patch, xy, subsample, r)
        x, y = self._find_landmark(img_patch, (x,y), 1, subsample)

        self.add_landmark(x,y)

    def get_patch(self, xy, sz=10):
        x, y = xy
        h, w = self.img.shape[:2]

        xmin = np.maximum(0, x-sz)
        xmax = np.minimum(x+sz, w)
        ymin = np.maximum(0, y-sz)
        ymax = np.minimum(y+sz, h)

        img_patch = self.img[ymin:ymax, xmin:xmax, :].mean(2) 
        return img_as_float(img_patch)

        
    @property
    def landmarks(self):
        if not self.xs or (np.isnan(self.xs)).all():
            return np.zeros((0,0))
        landmarks =  np.array(zip(self.xs, self.ys))
        landmarks = landmarks[~np.isnan(landmarks[:,0]),:]
        return landmarks


class RegistrationValidator:

    def __init__(self, target, ref, transform=None):
        self.im_orig = target
        self.im_reg = target
        self.im_ref = ref
        self.transform = None
        if transform:
            self.add_transform(transform)
        self.n_blks = ref.shape[0]/8, ref.shape[1]/8
        self._initialize_ui()
        self.region = None

    def _initialize_ui(self):
        self.ax = plt.subplot(111)
        bax_auto = plt.axes([0.1, 0.05, 0.15, 0.05])
        self._b_auto = Button(bax_auto, 'Auto')
        self._b_auto.on_clicked(self._on_auto)

        self._rs = RectangleSelector(self.ax, self._on_region_select)

    def _on_auto(self, event):
        transform = self.register()
        self.add_transform(transform)
        self._checkerboard()
        self.ax.figure.canvas.draw()

    def _on_region_select(self, eclick, erelease):
        xs = [int(eclick.xdata), int(erelease.xdata)]
        ys = [int(eclick.ydata), int(erelease.ydata)]
        xs.sort()
        ys.sort()
        self.region = xs+ys 


    def add_transform(self, transform):
        if self.transform is not None:
            self.transform += transform
        else:
            self.transform = transform
        self.im_reg = warp_int(self.im_orig, self.transform.inverse)


    def _checkerboard(self):
        im_reg = self.im_reg.copy()
        chboard_after = gen_checkerboard(im_reg, self.im_ref,
                                         self.n_blks)
        self.ax.imshow(chboard_after)

    def register(self):
        im = (self.im_reg.mean(2)).astype(np.uint8)
        ref = (self.im_ref.mean(2)).astype(np.uint8)
        if self.region is not None:
            xmin, xmax, ymin, ymax = self.region
            im = im[ymin:ymax, xmin:xmax]
            ref = ref[ymin:ymax, xmin:xmax]
        
        obj_func = lambda x: transform_and_compare(im, ref, x,
                                                  f_cmp=correlation_coefficient)
        x0 = np.array([0,0,1, 0])

        results = optimize.fmin_powell(obj_func, x0)
        logging.debug('Automatically found transform: %s' %
                      str(list(results)))
        trans = get_transform(results)
      
        return trans

    def set_landmarks(self, im1_coords, im2_coords):
        
        if self.transform:
            im1_coords = self.transform(im1_coords)
            
        self.ax.plot(im1_coords[:,0], im1_coords[:,1], 'o',ms=10)
        
        self.ax.plot(im2_coords[:,0], im2_coords[:,1], 'o', ms=10)



    def show(self):
        self._checkerboard()
        plt.show()

def mutual_information(img1, img2):
    K = kernel_mutualinformation(img1*1.,img2*1.)
    return -K()

def correlation_coefficient(img1, img2):
    return -np.corrcoef(img1.flatten(), img2.flatten())[0][1]

def mean_sq_diff(img1, img2):
    return ((img1-img2)**2).mean()

def img_diff(img1, img2):
    return (img1-img2)[img1>0].flatten()

def transform_and_compare(img, img_ref, params, f_cmp=mean_sq_diff):
    new_transform = get_transform(params)
    img_new = warp_int(img, new_transform.inverse)
    return f_cmp(img_new, img_ref)

def get_transform(x):
    dx, dy, s, rot = x
    return transform.AffineTransform(rotation=rot,
                                     scale=(s, s),
                                     translation=(dx, dy))
def landmark_error(coords, coords_ref, transf_factory):

    def _calc_error(params):
        t = transf_factory(params)
        err = (t(coords)-np.array(coords_ref)).flatten()
        return err

    return _calc_error

def register_landmarks(coords_target, coords_ref):

    err_func = landmark_error(coords_target, coords_ref, get_transform)

    transform_params,_ = optimize.leastsq(err_func, (0,0,1,0)) 
    est_transform = get_transform(transform_params)

    return est_transform

debug = False

if __name__ == "__main__":
    
    path = '/Users/bartosz/Desktop/TREE/registration/'
    fname1 = 'TREE_2011-10-20-16-18-02-220.jpg' 
    fname2 = 'TREE_2012-01-17-12-28-29_KO6L4705-274.jpg'
    #fname1 = 'TREE_2012-01-17-12-28-29_KO6L4705-274.jpg'
    #fname1='TREE_2011-12-18-12-12-08-257.jpg'
    
    logging.basicConfig(level=logging.DEBUG)
    img1 = io.imread(os.path.join(path, fname1))
    img2 = io.imread(os.path.join(path, fname2))
    # for testing only
    #simple_translation = get_transform((5,5,1,0.01))
    #img2 = warp_int(img1, simple_translation.inverse)


    plt.figure()
    reg_before = RegistrationValidator(img1, img2)
    if debug:
       fig_debug = plt.figure()
    reg_before.show()

    plt.figure()
    ax1=plt.subplot(121)
    im1_sel=LandmarkSelector(ax1, img1)
    ax1.set_title('Target')
    ax2=plt.subplot(122)
    ax2.set_title('Reference')
    im2_sel=LandmarkSelector(ax2, img2)

    ax_button = plt.axes([0.15, 0.05, 0.2, 0.1])
    button = Button(ax_button, 'Copy landmarks')
    def on_clicked(event):
        patch_size = 20
        im1_sel.remove_all_landmarks()
        ref_landmarks = im2_sel.landmarks
        for xy in ref_landmarks:
            im_patch = im2_sel.get_patch(xy,patch_size)
            im1_sel.find_landmark(im_patch,xy)
    button.on_clicked(on_clicked)
    plt.show()
    
    coords_reg = im1_sel.landmarks
    coords_ref = im2_sel.landmarks

    est_transform = register_landmarks(coords_reg, coords_ref)

    plt.figure()
    reg_after = RegistrationValidator(img1, img2, est_transform)
    reg_after.set_landmarks(coords_reg, coords_ref)
    
    if debug:
       fig_debug = plt.figure()

    reg_after.show()
