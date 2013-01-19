#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

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

def kernel_mutualinformation(img,patch, n=20):
    bins_x = np.linspace(img.min(), img.max(), n)
    bins_y = np.linspace(patch.min(), patch.max(), n)
    eps = np.finfo(np.float64).eps
    dx = (img.max()-img.min())*1./n 
    dy = (patch.max()-patch.min())*1./n 
    
    y = patch.flatten()
    ny, _ = np.histogram(y, bins_y, density=True)
    
    def _kernel(x=img):
        x = x.flatten()
        nx, _ = np.histogram(x, bins_x, density=True)
        nxy, _,_ = np.histogram2d(x,y,[bins_x, bins_y])
        nxy /= (np.sum(nxy)*dx*dy)
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
            m.remove()
        self.xs = []
        self.ys = []
        self.markers = []
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
                         kernel=kernel_mutualinformation)
        y, x = np.unravel_index(corr.argmax(), corr.shape)
        return int((x+0.5)*sub_sample)+l, int((y+0.5)*sub_sample)+b

    def find_landmark(self, img_patch, xy=None, r=100):
        subsample = 5
        
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
if __name__ == "__main__":
    
    path = '/Users/bartosz/Desktop/TREE/registration/'
    fname1 = 'TREE_2011-10-20-16-18-02-220.jpg' 
    fname2 = 'TREE_2012-01-17-12-28-29_KO6L4705-274.jpg'
    #fname1 = 'TREE_2012-01-17-12-28-29_KO6L4705-274.jpg'

    img1 = io.imread(os.path.join(path, fname1))
    img2 = io.imread(os.path.join(path, fname2))

    n_blks = img1.shape[0]/8, img1.shape[1]/8
    print img1.shape
    
    chboard_before = gen_checkerboard(img1, img2, n_blks)
    plt.figure()
    plt.imshow(chboard_before)
    plt.show()

    plt.figure()
    ax1=plt.subplot(121)
    im1_sel=LandmarkSelector(ax1, img1)
    ax1.set_title('Target')
    ax2=plt.subplot(122)
    ax2.set_title('Reference')
    im2_sel=LandmarkSelector(ax2, img2)

    from matplotlib.widgets import Button
    ax_button = plt.axes([0.15, 0.05, 0.2, 0.1])
    button = Button(ax_button, 'Copy landmarks')
    def on_clicked(event):
        im1_sel.remove_all_landmarks()
        ref_landmarks = im2_sel.landmarks
        for xy in ref_landmarks:
            im_patch = im2_sel.get_patch(xy,50)
            im1_sel.find_landmark(im_patch,xy)
    button.on_clicked(on_clicked)

    plt.show()
    coords_reg = im1_sel.landmarks
    coords_ref = im2_sel.landmarks

    err_func = landmark_error(coords_reg, coords_ref, get_transform)

    transform_params,_ = optimize.leastsq(err_func, (0,0,1,0)) 
    est_transform = get_transform(transform_params)

    from warps_py import warp_int 
    img_reg = warp_int(img1, est_transform.inverse)

    plt.figure()
    chboard_after = gen_checkerboard(img_reg.copy(), img2, n_blks)
    plt.imshow(chboard_after)
    plt.show()


