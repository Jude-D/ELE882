import numpy as np
from .. import filters


def denoise(img):
    '''Denoise an image with multiplicative noise.

    Parameters
    ----------
    img : numpy.ndarray
        colour input image with floating-point data type
    '''
    # Implement a denoising algorithm.  You may hard code the parameters since
    # this is only called by the test runner.  If you intend on processing in
    # colour then process the channels as three, separate images.  Otherwise
    # use skimage.colors.rgb2gray to convert to greyscale.

    image = img.astype(float)
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    denoised_r = filters.gaussian(r, 0.7)
    denoised_g = filters.gaussian(g, 0.7)
    denoised_b = filters.gaussian(b, 0.7)
    return np.stack((denoised_r, denoised_g, denoised_b), axis=2)
