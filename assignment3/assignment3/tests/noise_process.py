import abc

import numpy as np
from skimage.util import img_as_float


class NoiseProcess(abc.ABC):
    '''Helper class to quickly work with testing denoising functions.'''
    @abc.abstractmethod
    def apply(self, img):
        '''Apply the noise process to an image.'''
        pass

    def denoise(self, fn, img):
        '''Apply a denoising function against this noise process.'''
        img = img_as_float(img)
        noisy = self.apply(img)
        denoised = fn(noisy)
        return noisy, denoised


class AdditiveNoise(NoiseProcess):
    '''Adds additive noise to an image.

    Additive noise is any noise where ``In(x,y) = I(x,y) + n(x,y)``.  The noise
    used here is zero-mean Gaussian noise.
    '''
    def __init__(self, variance):
        self._variance = variance

    def apply(self, img):
        noise = np.random.normal(0, self._variance, img.shape)
        noisy = img + noise
        noisy[noisy < 0] = 0
        noisy[noisy > 1] = 1
        return noisy


class MultiplicativeNoise(NoiseProcess):
    '''Applies multiplicative noise to an image.

    Multiplicative noise is any noise where ``In(x, y) = (1 + n(x,y))*I(x,y)``.
    Here, ``n(x,y)`` is zero-mean Gaussian noise.
    '''
    def __init__(self, variance):
        self._variance = variance

    def apply(self, img):
        noise = np.random.normal(0, self._variance, img.shape)
        noisy = (1 + noise)*img
        noisy[noisy < 0] = 0
        noisy[noisy > 1] = 1
        return noisy


class ImpulsiveNoise(NoiseProcess):
    '''Applies impulsive noise to an image.

    Impulsive noise occurs when a pixel has a non-zero likelihood of being
    completely "on" (255) or "off" (0).  The noise is implemented by randomly
    setting some fraction of the image pixels to 255 or 0.
    '''
    def __init__(self, likelihood):
        self._p = likelihood

    def apply(self, img):
        num_pixels = int(np.floor(self._p*img.size))

        set_high = np.random.choice(img.size, num_pixels, replace=False)
        set_low = np.random.choice(img.size, num_pixels, replace=False)

        noisy = img.copy().flatten()

        noisy[set_high] = 1
        noisy[set_low] = 0
        noisy = noisy.reshape(img.shape)

        return noisy
