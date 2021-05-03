import numpy as np
from skimage.color import hsv2rgb, rgb2hsv
# from skimage.util import img_as_float


def adjust_saturation(img, amount):
    '''Adjust the amount of saturation in an image.

    Parameters
    ----------
    img : numpy.ndarray
        input colour image; it is converted into floating point if not already
        floating point
    amount : float
        value between -1 and 1 that controls the amount of saturation, where
        '+1' is maximum saturation and '-1' is completely desaturated

    Returns
    -------
    numpy.ndarray
        saturation-adjusted image (floating-point storage)

    Raises
    ------
    ValueError
        if the input image isn't 3-channel RGB or if 'amount' is on [-1, 1]
    '''
    if img.ndim != 3 or img.shape[2] != 3 or amount > 1 or amount < -1:
        raise ValueError("input image isn't 3-channel RGB or amount is wrong")
    img_hsv = rgb2hsv(img)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] + amount, 0, 1)
    return hsv2rgb(img_hsv)


def adjust_hue(img, amount):
    '''Adjust an image's hue by shifting it by a set amount of degrees.

    Parameters
    ----------
    img : numpy.ndarray
        input colour image; it is converted into floating point if not already
        floating point
    amount : float
        an angle, in degrees, representing the amount of hue shift

    Returns
    -------
    numpy.ndarray
        new image with the shifted hue

    Raises
    ------
    ValueError
        if the input image isn't 3-channel RGB
    '''
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("image isn't 3-channel rgb")
    img_hsv = rgb2hsv(img)
    img_hsv[:, :, 0] = np.mod((img_hsv[:, :, 0] + amount), 1.0)
    return hsv2rgb(img_hsv)


def to_monochrome(img, wr, wg, wb):
    '''Convert a colour image to monochrome using the provided weights.

    Parameters
    ----------
    img : numpy.ndarray
        input colour image; it is converted into floating point if not already
        floating point
    wr : float
        red channel weight
    wg : float
        green channel weight
    wb : float
        blue channel weight

    Returns
    -------
    numpy.ndarray
        grey scale image

    Raises
    ------
    ValueError
        if the input image is not colour or if any of the weights are negative
    '''
    if img.ndim != 3 or img.shape[2] != 3 or wr < 0 or wg < 0 or wb < 0:
        raise ValueError(
            'input image must be in color or weights must be positive')
    img = img.astype(float)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    grey = (wr * r + wg * g + wb * b)
    return np.clip(grey, 0, 1)
