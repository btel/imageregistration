#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from libc.math cimport ceil, floor
cimport numpy as np
import numpy as np

cdef inline unsigned char get_pixel2d(unsigned char* image, int rows, int cols, int r, int c,
                               char mode, unsigned char cval):
    """Get a pixel from the image, taking wrapping mode into consideration.

    Parameters
    ----------
    image : unsigned char array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : int
        Position at which to get the pixel.
    mode : {'C', 'W', 'R', 'N'}
        Wrapping mode. Constant, Wrap, Reflect or Nearest.
    cval : unsigned char
        Constant value to use for constant mode.

    Returns
    -------
    value : unsigned char
        Pixel value at given position.

    """
    if mode == 'C':
        if (r < 0) or (r > rows - 1) or (c < 0) or (c > cols - 1):
            return cval
        else:
            return image[r * cols + c]
    else:
        return image[coord_map(rows, r, mode) * cols + coord_map(cols, c, mode)]


cdef inline int coord_map(int dim, int coord, char mode):
    """
    Wrap a coordinate, according to a given mode.

    Parameters
    ----------
    dim : int
        Maximum coordinate.
    coord : int
        Coord provided by user.  May be < 0 or > dim.
    mode : {'W', 'R', 'N'}
        Whether to wrap or reflect the coordinate if it
        falls outside [0, dim).

    """
    dim = dim - 1
    if mode == 'R': # reflect
        if coord < 0:
            # How many times times does the coordinate wrap?
            if <int>(-coord / dim) % 2 != 0:
                return dim - <int>(-coord % dim)
            else:
                return <int>(-coord % dim)
        elif coord > dim:
            if <int>(coord / dim) % 2 != 0:
                return <int>(dim - (coord % dim))
            else:
                return <int>(coord % dim)
    elif mode == 'W': # wrap
        if coord < 0:
            return <int>(dim - (-coord % dim))
        elif coord > dim:
            return <int>(coord % dim)
    elif mode == 'N': # nearest
        if coord < 0:
            return 0
        elif coord > dim:
            return dim

    return coord
cdef inline unsigned char bilinear_interpolation(unsigned char* image, int rows, int cols,
                                          double r, double c, char mode,
                                          unsigned char cval):
    """Bilinear interpolation at a given position in the image.

    Parameters
    ----------
    image : unsigned char array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : double
        Position at which to interpolate.
    mode : {'C', 'W', 'R', 'N'}
        Wrapping mode. Constant, Wrap, Reflect or Nearest.
    cval : unsigned char
        Constant value to use for constant mode.

    Returns
    -------
    value : unsigned char
        Interpolated value.

    """
    cdef double dr, dc
    cdef int minr, minc, maxr, maxc

    minr = <int>floor(r)
    minc = <int>floor(c)
    maxr = <int>ceil(r)
    maxc = <int>ceil(c)
    dr = r - minr
    dc = c - minc
    cdef double top = (1 - dc) * get_pixel2d(image, rows, cols, minr, minc, mode, cval) \
          + dc * get_pixel2d(image, rows, cols, minr, maxc, mode, cval)
    cdef double bottom = (1 - dc) * get_pixel2d(image, rows, cols, maxr, minc, mode, cval) \
             + dc * get_pixel2d(image, rows, cols, maxr, maxc, mode, cval)
    return <unsigned char>ceil((1 - dr) * top + dr * bottom)

cdef inline void _matrix_transform(double x, double y, double* H, double *x_,
                                   double *y_):
    """Apply a homography to a coordinate.

    Parameters
    ----------
    x, y : double
        Input coordinate.
    H : (3,3) *double
        Transformation matrix.
    x_, y_ : *double
        Output coordinate.

    """
    cdef double xx, yy, zz

    xx = H[0] * x + H[1] * y + H[2]
    yy = H[3] * x + H[4] * y + H[5]
    zz =  H[6] * x + H[7] * y + H[8]

    x_[0] = xx / zz
    y_[0] = yy / zz


def warp_fast_int(np.ndarray image, np.ndarray H, output_shape=None, int order=1,
               mode='constant', int cval=0):
    """Projective transformation (homography).

    Perform a projective transformation (homography) of a
    floating point image, using bi-linear interpolation.

    For each pixel, given its homogeneous coordinate :math:`\mathbf{x}
    = [x, y, 1]^T`, its target position is calculated by multiplying
    with the given matrix, :math:`H`, to give :math:`H \mathbf{x}`.
    E.g., to rotate by theta degrees clockwise, the matrix should be

    ::

      [[cos(theta) -sin(theta) 0]
       [sin(theta)  cos(theta) 0]
       [0            0         1]]

    or, to translate x by 10 and y by 20,

    ::

      [[1 0 10]
       [0 1 20]
       [0 0 1 ]].

    Parameters
    ----------
    image : 2-D array
        Input image.
    H : array of shape ``(3, 3)``
        Transformation matrix H that defines the homography.
    output_shape : tuple (rows, cols)
        Shape of the output image generated.
    order : {0, 1}
        Order of interpolation::
        * 0: Nearest-neighbour interpolation.
        * 1: Bilinear interpolation (default).
        * 2: Biquadratic interpolation (default).
        * 3: Bicubic interpolation.
    mode : {'constant', 'reflect', 'wrap'}
        How to handle values outside the image borders.
    cval : string
        Used in conjunction with mode 'C' (constant), the value
        outside the image boundaries.

    """

    cdef np.ndarray[dtype=np.uint8_t, ndim=2, mode="c"] img = \
         np.ascontiguousarray(image, dtype=np.uint8)
    cdef np.ndarray[dtype=np.double_t, ndim=2, mode="c"] M = \
         np.ascontiguousarray(H)

    if mode not in ('constant', 'wrap', 'reflect', 'nearest'):
        raise ValueError("Invalid mode specified.  Please use "
                         "`constant`, `nearest`, `wrap` or `reflect`.")
    cdef char mode_c = ord(mode[0].upper())

    cdef int out_r, out_c
    if output_shape is None:
        out_r = img.shape[0]
        out_c = img.shape[1]
    else:
        out_r = output_shape[0]
        out_c = output_shape[1]

    cdef np.ndarray[dtype=np.uint8_t, ndim=2] out = \
         np.zeros((out_r, out_c), dtype=np.uint8)

    cdef int tfr, tfc
    cdef double r, c
    cdef int rows = img.shape[0]
    cdef int cols = img.shape[1]

    cdef unsigned char (*interp_func)(unsigned char *, int, int, double, double,
                               char, unsigned char)
    interp_func = bilinear_interpolation

    for tfr in range(out_r):
        for tfc in range(out_c):
            _matrix_transform(tfc, tfr, <double*>M.data, &c, &r)
            out[tfr, tfc] = interp_func(<unsigned char*>img.data, rows, cols, r, c,
                                        mode_c, cval)

    return out
