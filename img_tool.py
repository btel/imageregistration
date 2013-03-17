#!/usr/bin/env python
#coding=utf-8
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import Circle
from matplotlib import cm 
from matplotlib.widgets import Button
from matplotlib.widgets import RectangleSelector
from matplotlib import gridspec

import Tkinter as Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import ttk

import tkMessageBox
import tkFileDialog

import skimage
from skimage import img_as_ubyte, img_as_float
from skimage import io
from skimage import transform

from registration import corr

from scipy import optimize
from scipy import ndimage

from numpy.lib.stride_tricks import as_strided
from numpy import nan

import os
import csv
from itertools import cycle

from scipy.stats import gaussian_kde
from scipy.integrate import dblquad

from warps_py import warp_int 

import logging
from datetime import datetime

import shelve

def view_as_blocks(arr_in, block_shape):
    #modified from skimage.util.shape.view_as_blocks
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple')

    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")

    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length "
                         "as 'arr_in.shape'")

    arr_shape = np.array(arr_in.shape)

    # -- restride the array to build the block view
    arr_in = np.ascontiguousarray(arr_in)

    new_shape = tuple(arr_shape / block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)

    return arr_out

def gen_checkerboard(img1, img2, n_blks):
    blk_size = (img1.shape[0]/n_blks,
                img1.shape[1]/n_blks)

    img1 = img1.copy()
    img_check = view_as_blocks(img1, blk_size+(3,))
    img2_blks = view_as_blocks(img2, blk_size+(3,))

    img_check[:-1:2,:-1:2, :] = img2_blks[:-1:2,:-1:2,:]
    img_check[1::2,1::2, :] = img2_blks[1::2,1::2,:]
    return as_strided(img_check, img1.shape, img1.strides)

def imshow(im, ax):
    ax.imshow(im, interpolation='nearest')
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


class CrossHair(Circle):

    def __init__(self, xy, radius=5, **kwargs):

        Circle.__init__(self, xy, radius, **kwargs)
        self._path = Path([[-1, 0], [1,0], [0,-1], [0, 1]],
                         [Path.MOVETO, Path.LINETO, Path.MOVETO,
                          Path.LINETO])

    def set_center(self,xy):
        self.center = xy

def alert(msg, extra_msg=''):
        
    tkMessageBox.showinfo("Alert", msg)

    logging.info("%s (%s)" % (msg, extra_msg))

class LandmarkSelector:

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']

    def __init__(self, parent, subplot_spec, fname, title=''):

        self.xs = []
        self.ys = []
        self.markers = []
        
        self.fig = parent.fig
        self.parent = parent
        self.title = title

        self.imload(fname)

        self.color_iter = cycle(self.colors)

        self.gs = gridspec.GridSpecFromSubplotSpec(3, 3,
                                                   subplot_spec=subplot_spec,
                                                   height_ratios=[8,1,1],
                                                   wspace=0.05,
                                                   hspace=0.05
                                                  )

        self.ax = self.fig.add_subplot(self.gs[0,:])
        self.ax.set_title("%s\n(%s)" % (title, fname), size=10)
        #self.fig.add_subplot(self.ax)
        self.marker_radius = 100.

        self.picker_radius = 100
        self.zoom_factor = 5.
        self._update_image()

        self._dragged = None
        self.zoom_axes = None
                 
        self._init_ui()

    def _update_image(self):
        self.ax.imshow(self.img, interpolation='nearest')
        logging.info("Loaded image %s (%s)" % (self.fname,
                                               self.title))
        self.ax.set_title("%s\n(%s)" % (self.title, self.fname), size=10)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.canvas.draw()


    def _show_cursor(self, show):
        root = self.fig.canvas._tkcanvas.winfo_toplevel()
        if show:
            root.config(cursor='arrow')
        else:
            root.config(cursor='none')

    def _init_ui(self):
        self.click_ev = self.fig.canvas.mpl_connect('button_press_event',
                                       self._onclick)
        self.motion_ev = self.fig.canvas.mpl_connect('motion_notify_event',
                                       self._onmotion)
        self.release_ev = self.fig.canvas.mpl_connect('button_release_event',
                                       self._onrelease)

        ax_load_img = self.fig.add_subplot(self.gs[1,:])
        ax_load = self.fig.add_subplot(self.gs[2,0])
        ax_save = self.fig.add_subplot(self.gs[2,1])
        ax_reset = self.fig.add_subplot(self.gs[2,2])

        self.fig.add_subplot(ax_load_img)
        self.fig.add_subplot(ax_load)
        self.fig.add_subplot(ax_save)
        self.fig.add_subplot(ax_reset)
        
        self._load_img_button = Button(ax_load_img, 'Load image')
        self._load_img_button.on_clicked(self._on_load_image)

        self._load_button = Button(ax_load, 'Load LMs')
        self._load_button.on_clicked(self._on_load_landmarks)
        
        self._save_button = Button(ax_save, 'Save LMs')
        self._save_button.on_clicked(self._on_save_landmarks)
        
        self._reset_button = Button(ax_reset, 'Reset LMs')
        self._reset_button.on_clicked(self._on_reset_landmarks)

    def _on_load_image(self, event):
        fname = tkFileDialog.askopenfilename()
        if not fname:
            return
        self.imload(fname)
        self._update_image()
        self.remove_all_landmarks()
        self.parent.update()

    def _on_load_landmarks(self, event):
        self.load_landmarks()
    
    def _on_save_landmarks(self, event):
        self.save_landmarks()
    
    def _on_reset_landmarks(self, event):
        self.remove_all_landmarks()
        
    def imload(self, fname):
        try:
            img = io.imread(os.path.join(img_path, fname))
        except IOError as e:
            msg = e.message
            logging.warning('Loading image %s from %s failed: %s' %
                            (fname, img_path, msg))
            return
        path, core = os.path.split(fname)
        self.fname = core
        self.img = img

    def _onclick(self, event):
        if event.inaxes==self.ax:
            if event.button==1:
                self._left_click(event)
            elif event.button==3:
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
        self._show_cursor(False)
    
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
        self._show_cursor(True)

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
        m.set_center((x,y))
        
        self.ax.figure.canvas.draw()

        return m

    def _add_patch(self, xy, color):
        patch = CrossHair(xy, self.marker_radius, 
                               edgecolor=color, 
                              facecolor='none')
        self.ax.add_patch(patch)
        return patch
    
    def add_landmark(self, x,y):
        
        try:
            i = self.xs.index(nan)
            patch = self.update_landmark(i, x,y)
            self.ax.add_patch(patch)
            self.ax.figure.canvas.draw()
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
    
    def _find_landmark(self, img_patch, xy, sub_sample, r, debug=False):
        
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

        y, x = np.unravel_index(corr.argmax(), corr.shape)
        
        if debug:
            fig_debug = plt.figure()
            ax1 = fig_debug.add_subplot(221)
            ax1.imshow(corr, cmap=cm.gray)
            ax1.plot([x], [y], 'o')
            ax1.set_title('correlation')
            ax2 = fig_debug.add_subplot(222)
            ax2.imshow(img_float, cmap=cm.gray)
            ax2.imshow(corr, cmap=cm.hot, alpha=0.2)
            ax2.plot([x], [y], 'o')
            ax2.set_title('target image')
            ax3 = fig_debug.add_subplot(223)
            ax3.imshow(img_patch, cmap=cm.gray, interpolation='nearest')
            ax3.set_title('patch')
            fig_debug.canvas.draw()
            fig_debug.show()
            raw_input('press enter to continue')
            plt.close(fig_debug)

        return int((x+0.5)*sub_sample)+l, int((y+0.5)*sub_sample)+b

    def find_landmark(self, img_patch, xy=None, r=150):
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

    def save_landmarks(self):

        db = shelve.open('landmarks.db')

        db[str(self.fname)] = self.landmarks

        db.close()

        logging.info('Saved landmarks for %s' % self.fname)

    def load_landmarks(self):

        db = shelve.open('landmarks.db')

        try:
            landmarks = db[str(self.fname)]
        except KeyError:
            logging.debug('Landmarks for %s not found' % self.fname)
            return

        db.close()
        
        self.remove_all_landmarks()
        for x,y in landmarks:
            self.add_landmark(x,y)


class RegistrationToolbar:

    def __init__(self, main_window):
        self.root = main_window.root
        self.window = main_window
        self.frame = Tk.Frame(master=self.root)
        self._initalize_ui()

    def _initalize_ui(self):
        self._reg_combo_label = ttk.Label(self.frame, 
                                           text='Saved transforms:')
        self._reg_params = Tk.StringVar()
        self._reg_params.set('Current')
        self._reg_combo = ttk.Combobox(self.frame,
                               textvariable=self._reg_params)
        self._reg_combo.state(['readonly'])
        self._reg_combo.bind('<<ComboboxSelected>>', 
                             self._reg_params_selected)

        self._n_blks_label = ttk.Label(self.frame, 
                                      text='Checkerboard size:')
        self._n_blks_var = Tk.StringVar()
        self._n_blks_var.set(str(self.window.n_blks))
        self._n_blks_spinbox = Tk.Spinbox(self.frame,
                                           from_=1.0,
                                           to=99.0,
                                           width=2,
                                           textvariable=self._n_blks_var)
        self._n_blks_var.trace("w", self._on_nblks_changed)

        self._reg_combo_label.pack(side=Tk.LEFT)
        self._reg_combo.pack(side=Tk.LEFT)
        self._n_blks_label.pack(side=Tk.LEFT)
        self._n_blks_spinbox.pack(side=Tk.LEFT)

    def _on_nblks_changed(self, *args):
        n_blks = self._n_blks_var.get()
        self.window.set_nblks(int(n_blks))

    def set_transforms(self, descriptions):
        self._reg_combo['values'] = descriptions

    def select_transform(self, value):
        self._reg_params.set(value)

    def _reg_params_selected(self, event):
        id = self._reg_combo.current()
        self.window.select_saved_transform(id)
    
    def update(self):
        self.frame.pack(side=Tk.BOTTOM, fill=Tk.X, expand=0)


class RegistrationValidator:

    def __init__(self, target, ref):
        self.im1_sel = target
        self.im2_sel = ref
        self.im_reg = self.im1_sel.img
        
        self.transform = transform.AffineTransform()
       
        self.n_blks = 8
        self._coords = []

        self._check_file()

        self._open_figure()
        self.update()


    def _open_figure(self):
        self.root = Tk.Toplevel()
        self.root.title('Registration')
        self.fig = Figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        self.toolbar = RegistrationToolbar(self)
        self.toolbar.update()

        self._fig_is_open=True
        self._initialize_mpl_ui()
        self.region = None

    def set_nblks(self, n):
        self.n_blks = n
        self._checkerboard()

    def _initialize_mpl_ui(self):
        self.ax = self.fig.add_subplot(111)
        bax_auto = self.fig.add_axes([0.1, 0.05, 0.15, 0.05])
        self._b_auto = Button(bax_auto, 'Auto')
        self._b_auto.on_clicked(self._on_auto)
        
        bax_save = self.fig.add_axes([0.25, 0.05, 0.15, 0.05])
        self._b_save = Button(bax_save, 'Save')
        self._b_save.on_clicked(self._on_save)
        
        bax_reset = self.fig.add_axes([0.40, 0.05, 0.15, 0.05])
        self._b_reset = Button(bax_reset, 'Clear sel')
        self._b_reset.on_clicked(self._on_reset_region)

        self.fig.canvas.mpl_connect('close_event', self._on_close)

    def _on_close(self, event):
        self._fig_is_open=False

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
        self.ax.set_xlim(xs)
        self.ax.set_ylim(ys[::-1])
        self.ax.figure.canvas.draw()

    def _on_reset_region(self, event):
        self.region=None
        self.ax.set_xlim(0,self.im_reg.shape[1])
        self.ax.set_ylim(self.im_reg.shape[0],0)
        self.ax.figure.canvas.draw()
    
    def show_transform(self, transform_, date_='new'):
        logging.info('Showing transform: ' +
                     self._fmt_transform(transform_, date_))
        self._transform_date = date_
        self._selected_transform = transform_
        self.im_reg = warp_int(self.im1_sel.img, self._selected_transform.inverse)
        self._checkerboard()

    def reset_transform(self):
        self._transform_date = 'new'
        self.transform = transform.AffineTransform()
        self._selected_transform = self.transform 
        self.toolbar.select_transform('Current')
        self.im_reg = self.im1_sel.img
        self._checkerboard()

    def add_transform(self, transform):
        if self.transform is not None:
            self.transform += transform
        else:
            self.transform = transform
        self.im_reg = warp_int(self.im1_sel.img, self.transform.inverse)
        self._checkerboard()

    @property
    def transform_description(self):
        description = self._fmt_transform(self._selected_transform,
                                         self._transform_date)
        return description

    def _checkerboard(self):
        im_reg = self.im_reg
        im_ref = self.im2_sel.img
        chboard_after = gen_checkerboard(im_reg, im_ref,
                                         self.n_blks)
        self.ax.imshow(chboard_after, interpolation='nearest')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.ax.set_title(self.transform_description)

        self.fig.canvas.draw()

    def _show_landmarks(self):
        coords = []
        if len(self.im1_sel.landmarks)>0:
            im1_coords = self.transform(self.im1_sel.landmarks)
            coords.append(im1_coords)
        if len(self.im2_sel.landmarks)>0:
            coords.append(self.im2_sel.landmarks)
        for c in coords:
            self.ax.plot(c[:,0], c[:,1], '+',ms=10)

    def _parse_transform(self, p):
        p = map(float, p)
        matrix = np.array(p).reshape(3,3)
        trans = transform.AffineTransform(matrix=matrix)
        return trans
   
    def load_transforms(self):
        self._transforms_list = []
        with file('transforms.txt', 'r') as fid:
            csv_reader = csv.reader(fid)
            for row in csv_reader:
                date, target_img, ref_img = row[:3]
                if (target_img == self.im1_sel.fname and 
                       ref_img == self.im2_sel.fname):

                    transform_params = row[3:]
                    trans = self._parse_transform(transform_params)
                    self._transforms_list.append((date, trans))

    def _on_save(self, event):
        transform = self.transform

        fname_target = self.im1_sel.fname
        fname_ref = self.im2_sel.fname
        
        
        if transform is None:
            alert('Transform not defined. Nothing was saved!')
            logging.warning('Save attempted without transform defined'
                          ' (%s - %s)' % (fname_target, fname_ref))
            return

        params = map(str, list(transform._matrix.flatten()))
        now = datetime.now()

        timestamp = now.strftime("%d/%m/%Y %H:%M") 

        line = ",".join([timestamp, 
                        fname_target, 
                        fname_ref]+
                        params)+'\n'
        
        try:
            with file('transforms.txt', 'r') as fid_read:
                if line in fid_read:
                    alert('Transform already saved', line)
                    return
        except IOError as e:
		    pass
        with file('transforms.txt', 'a') as fid:
            fid.write(line)
        alert('Transform %s saved' % self._fmt_transform(transform,
                                                         timestamp))
        self.update_transform_list()

    def register(self):
        im = (self.im_reg.mean(2)).astype(np.uint8)
        ref = (self.im2_sel.img.mean(2)).astype(np.uint8)
        if self.region is not None:
            xmin, xmax, ymin, ymax = self.region
            im = im[ymin:ymax, xmin:xmax]
            ref = ref[ymin:ymax, xmin:xmax]
        
        obj_func = lambda x: transform_and_compare(im, ref, x,
                                                  f_cmp=correlation_coefficient)
        x0 = np.array([0,0,1, 0])

        results = optimize.fmin_powell(obj_func, x0)
        logging.info('Automatically found transform: %s' %
                      str(list(results)))
        trans = get_transform(results)
      
        return trans

    def landmark_register(self):
        coords_reg = self.im1_sel.landmarks
        coords_ref = self.im2_sel.landmarks
        if len(coords_reg)==0 or len(coords_ref)==0:
            return
        self.transform = register_landmarks(coords_reg, coords_ref)
        self.show_transform(self.transform)

    def _fmt_transform(self, t, date=''):
        x, y = t.translation
        s_x, s_y = t.scale
        rot = 180*t.rotation/np.pi
        
        s = "x=%d, y=%d, s=%.2f, rot=%.2f" % (x,y, s_x, rot)

        if date:
            s += " (%s)" % date
        
        return s


    def get_transform_descriptions(self):
        descr = [date for date, t in self._transforms_list]
        return descr
            

    def update_transform_list(self):
        self.load_transforms()
        descr = self.get_transform_descriptions()
        self.toolbar.set_transforms(['Current'] + descr)

    def select_saved_transform(self, i):
        if i == 0:
            self.show_transform(self.transform)
        else:
            date, trans = self._transforms_list[i-1]
            self._transform_date = date
            self.show_transform(trans, date)

    def _check_file(self):
        try:
            with file("transforms.txt", 'a') as fid:
                pass
        except IOError as e:
            print e
            logging.error("Could not open transforms.txt in %s (%s)" %
                        (os.path.abspath(os.curdir), e))
            raise


    def update(self):
        self.ax.cla()
        self.reset_transform()
        self.update_transform_list()
        self.landmark_register()
        self._show_landmarks()
        self._checkerboard()
        self._rs = RectangleSelector(self.ax, self._on_region_select)
        self.fig.canvas.draw()

class Application:

    def __init__(self, img1, img2):
        self.root = Tk.Tk()
        self.root.title('Landmark selection')
        self.fig = Figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        
        gs = gridspec.GridSpec(2,2, height_ratios=[8,1],wspace=0.05,
                              hspace=0.05, left=0.05, right=0.95,
                               top=0.95)
        self.im1_sel = LandmarkSelector(self, gs[0,0], img1, 'Target')
        self.im2_sel = LandmarkSelector(self, gs[0,1], img2, 'Reference')

        self.comparator = RegistrationValidator(
                                                self.im1_sel, 
                                                self.im2_sel)
        
        self._gs = gs
        self._init_panel()
        self.canvas.draw()

    def _on_copy(self, event):
        patch_size = 50
        im1_sel = self.im1_sel
        im2_sel = self.im2_sel
        im1_sel.remove_all_landmarks()
        ref_landmarks = im2_sel.landmarks
        for xy in ref_landmarks:
            im_patch = im2_sel.get_patch(xy,patch_size)
            im1_sel.find_landmark(im_patch,xy)

    def _on_register(self, event):
        self.comparator.im1_sel = self.im1_sel
        self.comparator.im2_sel = self.im2_sel

        self.comparator.update()

    def update(self):
        self.comparator.update()

    def _init_panel(self):
        
        self._gs_panel = gridspec.GridSpecFromSubplotSpec(1, 6,
                                                          wspace=0.05,
                                                          hspace=0.05,
                                                   subplot_spec=self._gs[1,:])
        ax_copy = self.fig.add_subplot(self._gs_panel[0])
        ax_register = self.fig.add_subplot(self._gs_panel[1])
        
        self.fig.add_subplot(ax_copy)
        self.fig.add_subplot(ax_register)

        self._copy_button = Button(ax_copy, 'Copy\nlandmarks')
        self._copy_button.on_clicked(self._on_copy)
        
        self._register_button = Button(ax_register, 'Register')
        self._register_button.on_clicked(self._on_register)
        
    def run(self):
        Tk.mainloop()


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
    logging.info('Transform parameters estimated from landmarks: '
                     + str(transform_params))
    est_transform = get_transform(transform_params)

    return est_transform

debug = True

if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', '-d', action='store_true')

    args = parser.parse_args()

    img_path = '.'
    fname1 = 'TREE_2011-10-20-16-18-02-220.jpg' 
    fname2 = 'TREE_2012-01-17-12-28-29_KO6L4705-274.jpg'

    fname1='TREE_2011-09-26-08-21-03-192.jpg'
    fname2='TREE_2011-09-18-07-57-27-184.jpg'
   
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.DEBUG, filename='img_tool.log')

    # for testing only
    #simple_translation = get_transform((5,5,1,0.01))
    #img2 = warp_int(img1, simple_translation.inverse)


    app = Application(fname1, fname2)
    app.run()
