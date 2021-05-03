import numpy as np


def rgb2grey(image):
    '''Convert a RGB colour image into a greyscale image.

    The image is converted into RGB by taking a weighted sum of the three colour
    channels.  I.e.,

    .. math::

        I(x,y) = 0.299 R(x,y) + 0.587 G(x,y) + 0.114 B(x,y).

    The image should be converted to floating point prior to the calculation so
    that it's on [0, 1].  After generating the greyscale image, it should be
    converted back to 8bpc.

    Parameters
    ----------
    image : numpy.ndarray
        a 3-channel, RGB image

    Returns
    -------
    numpy.ndarray
        a single channel, monochome image derived from the original

    Raises
    ------
    ValueError
        if the image is already greyscale or if the input image isn't 8bpc
    '''

    # Take in the RGB image
    if image.ndim != 3:
        raise ValueError('Image is already greyscale')
    if image.dtype != np.uint8:
        raise ValueError('Can only support 8-bit images.')

    # Convert the image from a RGB image to a Grey Image
    image = image.astype(float)/255
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    grey = (0.299 * r + 0.587 * g + 0.114 * b)*255
    #grey = np.clip(grey, 0, 255)
    #grey = grey.astype(np.uint8)*255
    values = np.array(grey, dtype=np.uint8)
    return values
    # raise NotImplementedError('Implement this function/method.')


def grey2rgb(image):
    '''Pseudo-convert a greyscale image into an RGB image.

    This will make an greyscale image appear to be RGB by duplicating the
    intensity channel three times.

    Parameters
    ----------
    image : numpy.ndarray
        a greyscale image

    Returns
    -------
    numpy.ndarray
        a three-channel, RGB image

    Raises
    ------
    ValueError
        if the input image is already RGB or if the image isn't 8bpc
    '''
    if image.ndim != 2:
        raise ValueError('Image is already in color')
    if image.dtype != np.uint8:
        raise ValueError('Can only support 8-bit images.')
    return np.stack(
        (image, image, image), axis=2)
    # raise NotImplementedError('Implement this function/method.')
