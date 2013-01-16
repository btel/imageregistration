#!/usr/bin/env python
#coding=utf-8

from warps import warp_fast_int
import numpy as np

from skimage import data


def warp_int(image, inverse_map,  map_args={}, output_shape=None, order=1,
         mode='constant', cval=0.):
    
    matrix = np.linalg.inv(inverse_map.im_self._matrix)
    if matrix is not None:
        # transform all bands
        dims = []
        for dim in range(image.shape[2]):
            dims.append(warp_fast_int(image[..., dim], matrix,
                        output_shape=output_shape,
                        order=order, mode=mode, cval=cval))
        out = np.dstack(dims)
    return out

if __name__ == '__main__':
    
    from skimage import img_as_ubyte
    img = img_as_ubyte(data.lena())
    from img_tool import get_transform
    import matplotlib.pyplot as plt

    t = get_transform((10,10,1,0))

    img_new = warp_int(img, t.inverse)
    plt.imshow(img_new)
    plt.show()
