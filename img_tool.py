#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

import skimage
from skimage import img_as_ubyte
from skimage import io
from skimage.util.shape import view_as_blocks
from skimage import transform
from scipy import optimize

from numpy.lib.stride_tricks import as_strided
from numpy import nan

import os
from itertools import cycle


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
        self._dragged.set_radius(self.marker_radius*1./zoom)
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


    def update_landmark(self, i, x, y):

        self.xs[i] = x
        self.ys[i] = y
       
        m = self.markers[i]
        color = m.get_facecolor()
        
        try:
            m.remove()
        except ValueError:
            #marker was already removed
            pass

        new_marker = self._add_patch((x,y), color)
        new_marker.set_radius(self.marker_radius*1./self.zoom_factor)
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

        self.xs.append(x)
        self.ys.append(y)
        
        color = next(self.color_iter)

        patch = self._add_patch((x,y), color)
        self.markers.append(patch)

        self.ax.figure.canvas.draw()

        return patch

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


