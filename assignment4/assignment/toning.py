import numpy as np
from skimage.color import hsv2rgb, rgb2hsv
# from skimage.util import img_as_float
from . import adjustment


def single_tone(img, hue, saturation, amount=1.0):
    '''Apply a colour tone to an image.

    The toning process is the same for both greyscale and colour image.  Colour
    images are simply converted into HSV.  A greyscale image is "converted" to
    colour by replicating it three times before converting it into HSV.

    Parameters
    ----------
    img : numpy.ndarray
        a colour or greyscale input image; the data type will be converted to
        float if it is not already
    hue : float
        the value of the tone's hue; must be an angle on [0, 360]
    saturation : float
        the tone's saturation; must be on [0, 1]
    amount : float, optional
        a value between 0 and 1 on how much of the tone to apply to the original
        image; the default is 1.0, or completely replace all colour

    Returns
    -------
    numpy.ndarray
        the colour-toned image

    Raises
    ------
    ValueError
        if any of the input values are invalid
    '''
    if img.ndim != 2 and img.ndim != 3:
        raise ValueError('Image is not greyscale or colour')
    if hue < 0 or hue > 360:
        raise ValueError("Hue is invalid")
    if saturation < 0 or saturation > 1:
        raise ValueError("Saturation is invalid")
    if amount < 0 or amount > 1:
        raise ValueError("Amount is invalid")
    hue /= 360
    if img.ndim == 3:
        img_new = rgb2hsv(img)
        img_new[:, :, 0] = hue
        img_new[:, :, 1] = saturation
        return np.clip((1 - amount) * (img) + amount * hsv2rgb(img_new), -1, 1)
    else:
        img_new = np.stack(
            [np.full(img.shape, hue), np.full(img.shape, saturation), img], axis=2)
        return np.clip((1 - amount) * np.stack([img, img, img], axis=2)
                       + amount * hsv2rgb(img_new), -1, 1)


def split_tone(img, highlight, shadow):
    '''Apply split toning to an image.

    Parameters
    ----------
    img : numpy.ndarray
        input RGB image
    highlight : (hue, saturation)
        the highlight tone
    shadow : (hue, saturation)
        the shadow tone

    Returns
    -------
    numpy.ndarray
        the split-toned image
    '''
    highlight = single_tone(img, highlight[0], highlight[1])
    shadow = single_tone(img, shadow[0], shadow[1])
    g = adjustment.to_monochrome(img, 0.299, 0.587, 0.114)
    g_max = np.max(g)
    m = g/g_max if g_max != 0 else np.zeros(g.shape)
    m = np.stack([m, m, m], axis=2)
    return m * highlight + (1-m) * shadow
