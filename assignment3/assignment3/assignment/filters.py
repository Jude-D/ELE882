import numpy as np
import math
from scipy import ndimage


def _convolve(img, kernel):
    '''Convenience method around ndimage.convolve.

    This calls ndimage.convolve with the boundary setting set to 'nearest'.  It
    also performs some checks on the input image.

    Parameters
    ----------
    img : numpy.ndarray
        input image
    kernel : numpy.ndarray
        filter kernel

    Returns
    -------
    numpy.ndarray
        filter image

    Raises
    ------
    ValueError
        if the image is not greyscale
    TypeError
        if the image or filter kernel are not a floating point type
    '''
    if img.ndim != 2:
        raise ValueError('Only greyscale images are supported.')

    if img.dtype != np.float32 and img.dtype != np.float64:
        raise TypeError('Image must be floating point.')

    if kernel.dtype != np.float32 and img.dtype != np.float64:
        raise TypeError('Filter kernel must be floating point.')

    return ndimage.convolve(img, kernel, mode='nearest')


def moving_average(img, width):
    '''Filter an image using a moving-average ("box") filter.

    A box filter is the average of all pixel values within an NxN neighbourhood.
    The filter is implemented as a two-pass, separable kernel.

    Parameters
    ----------
    img : numpy.ndarray
        input image
    width : int
        width of the filter kernel, i.e. 'N'; must be positive and an odd-valued

    Returns
    -------
    numpy.ndarray
        filtered image

    Raises
    ------
    ValueError
        if the width is even, zero or negative
    '''
    if (width % 2) == 0:
        raise ValueError('Width is even')
    elif width <= 0:
        raise ValueError('Width cannot be less than or equal to 0')

    width_kernel = np.ones((1, width), dtype=np.float32)/width
    height_kernel = np.ones((width, 1), dtype=np.float32)/width

    convolve_width = _convolve(img, width_kernel)
    convolve_height = _convolve(convolve_width, height_kernel)
    return convolve_height


def gaussian(img, sigma):
    '''Filter an image using a Gaussian kernel.

    The Gaussian is implemented internally as a two-pass, separable kernel.

    Note
    ----
    The kernel is scaled to ensure that its values all sum up to '1'.  The
    slight truncation means that the filter values may not actually sum up to
    one.  The normalization ensures that this is consistent with the other
    low-pass filters in this assignment.

    Parameters
    ----------
    img : numpy.ndarray
        a greyscale image
    sigma : float
        the width of the Gaussian kernel; must be a positive, non-zero value

    Returns
    -------
    numpy.ndarray
        the Gaussian blurred image; the output will have the same type as the
        input

    Raises
    ------
    ValueError
        if the value of sigma is negative
    '''
    if (sigma <= 0.01):
        raise ValueError("Sigma cannot be less than 0")

    kernel_width = 6 * sigma
    N = max(math.ceil(kernel_width), 3)
    print(N)

    if (N % 2) == 0:
        N += 1

    vertical_kernel = np.ones((N, 1), dtype=np.float32)
    for i in range(0, N):
        vertical_kernel[i, 0] = (1/math.sqrt(2*math.pi*sigma**2)) * \
            math.exp(-(i-math.floor(0.5*N))**2/(2*sigma**2))
    horizontal_kernel = np.transpose(vertical_kernel)
    mask_sum = (vertical_kernel*horizontal_kernel).sum()
    vertical_conv = _convolve(img, vertical_kernel)
    horizontal_kernel = _convolve(vertical_conv, horizontal_kernel)
    gaus = (1/mask_sum)*horizontal_kernel
    print(gaus)
    return gaus


def laplacian(img):
    '''Filter an image using a Laplacian kernel.

    The Laplacian kernel used by this function is::

        [  0 -1  0 ]
        [ -1  4 -1 ]
        [  0 -1  0 ]

    Parameters
    ----------
    img : numpy.ndarray
        a greyscale image

    Returns
    -------
    numpy.ndarray
        the image after convolving with the Laplacian kernel
    '''
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return _convolve(img, kernel)
