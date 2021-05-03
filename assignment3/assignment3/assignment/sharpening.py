from . import filters
import numpy as np


def unsharp_masking(img, gain, sigma):
    '''Sharpen an image via unsharp masking.

    Parameters
    ----------
    img : numpy.ndarray
        input image
    gain : float
        strength of the sharpening
    sigma : float
        the width of the Gaussian blur kernel

    Returns
    -------
    numpy.ndarray
        the sharpened image

    Raises
    ------
    ValueError
        if the gain is negative
    '''
    if gain < 0:
        raise ValueError('Gain cannot be negative')
    return np.clip(img + gain*(img - filters.gaussian(img, sigma)), 0, 1)


def laplacian(img, gain):
    '''Sharpen an image via Laplacian sharpening.

    Parameters
    ----------
    img : numpy.ndarray
        input image
    gain : float
        strength of the sharpening

    Returns
    -------
    numpy.ndarray
        the sharpened image

    Raises
    ------
    ValueError
        if the gain is negative
    '''
    if gain < 0:
        raise ValueError('Gain cannot be negative')

    return np.clip(img + gain * filters.laplacian(img), 0, 1)
