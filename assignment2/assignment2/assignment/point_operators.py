import numpy as np


def apply_lut(img, lut):
    '''Apply a look-up table to an image.

    The look-up table can be be used to quickly adjust the intensities within an
    image.  For colour images, the same LUT can be applied equally to each
    colour channel.

    Parameters
    ----------
    img : numpy.ndarray
        a ``H x W`` greyscale or ``H x W x C`` colour 8bpc image
    lut : numpy.ndarray
        a 256-element, 8-bit array

    Returns
    -------
    numpy.ndarray
        a new ``H x W`` or ``H x W x C`` image derived from applying the LUT

    Raises
    ------
    ValueError
        if the LUT is not 256-elements long
    TypeError
        if either the LUT or images are not 8bpc
    '''
    if lut.shape != (256,):
        raise ValueError("LUT is not 256-elements long")
    if img.dtype != np.uint8 or lut.dtype != np.uint8:
        raise TypeError('Can only work on 8-bit images.')
    if 2 <= img.ndim <= 3:
        return lut[img]
    else:
        raise ValueError("image dimensions not supported")


def adjust_brightness(offset):
    '''Generate a LUT to adjust the image brightness.

    Parameters
    ----------
    offset : int
        the amount to offset brightness values by; this may be negative or
        positive

    Returns
    -------
    numpy.ndarray
        a 256-element LUT that can be provided to ``apply_lut()``
    '''
    return np.clip(np.arange(0, 256) + offset, 0, 255).astype(np.uint8)


def adjust_contrast(scale, hist):
    '''Generate a LUT to adjust contrast without affecting brightness.

    Parameters
    ----------
    scale : float
        the value used to adjust the image contrast; a value greater than 1 will
        increase constrast while a value less than 1 will reduce it
    hist : numpy.ndarray
        a 256-element array containing the image histogram, which is used to
        calculate the image brightness

    Returns
    -------
    numpy.ndarray
        a 256-element LUT that can be provided to ``apply_lut()``

    Raises
    ------
    ValueError
        if the histogram is not 256-elements or if the scale is less than zero
    '''
    if hist.shape != (256,):
        raise ValueError("Histogram is not 256 elements")

    brightness = np.sum(hist * np.arange(0, 256, dtype=float)
                        ) / np.sum(hist, dtype=float)
    contrast = np.arange(0, 256, dtype=float)*scale + (1-scale)*brightness
    return np.clip(contrast, 0, 255).astype(np.uint8)


def adjust_exposure(gamma):
    '''Generate a LUT that applies a power-law transform to an image.

    Parameters
    ----------
    gamma : float
        the exponent in the power-law transform; must be a positive value

    Returns
    -------
    numpy.ndarray
        a 256-element LUT that can be provided to ``apply_lut()``

    Raises
    ------
    ValueError
        if ``gamma`` is negative
    '''
    if gamma < 0:
        raise ValueError("gamma is negative")

    return (np.power(np.arange(0, 256, dtype=float)/255.0, gamma)*255.0).astype(np.uint8)


def log_transform():
    '''Generate a LUT that applies a log-transform to an image.

    Returns
    -------
    numpy.ndarray
        a 256-element LUT that can be provided to ``apply_lut()``
    '''
    c = 106
    log = np.log10(np.arange(0, 256, dtype=float) + 1)
    return np.clip((log * c), 0, 255).astype(np.uint8)
