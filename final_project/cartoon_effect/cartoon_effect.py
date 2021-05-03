import numpy as np
import math
from scipy import ndimage
from skimage.color import rgb2grey
from skimage.morphology import label


class CartoonEffect:

    def apply(self, img, sigmax, row, T, sigmab, N):

        if img.ndim != 3 and img.ndim != 2:
            raise ValueError('Image is not greycale or rgb')
        elif img.ndim == 3:
            'convert to greyscale if image is rgb'
            Gimg = rgb2grey(img)
        else:
            Gimg = img
        'step 1 and 2, solve for gaussian blur on image'
        Gimg = self.gaussian(Gimg, sigmab)
        'step 3 and 4, solve for min and max of image'
        Imin = np.min(Gimg)
        Imax = np.max(Gimg)
        'step 5 and step 6, solve for bins[i]'
        bins = np.clip((Imax-Imin)/N*np.arange(N+1) + Imin, 0, 255)
        'step 7, allocate the image L'
        L = np.zeros(img.shape[:2], dtype=int)
        'step 8, fill in L'
        for i in range(N):
            if i != N-1:
                mask = np.logical_and(Gimg >= bins[i], Gimg < bins[i+1])
            else:
                mask = Gimg >= bins[i]
            L[mask] = i
        'step 9, allocate the image S'
        S = np.zeros(img.shape)
        'step 10,'
        C = np.zeros(img.shape, dtype=int)
        if img.ndim == 3:
            L = label(L)
            print(L.shape)
            for i in range(255):
                Li = L == i
                Li_stacked = np.stack((Li, Li, Li), axis=2)
                mean_colour = np.mean(img, where=Li_stacked, axis=(0, 1))
                C[Li] = mean_colour
        else:
            C /= N
        return C * self.XDoG(img, sigmax, row, T)

    def XDoG(self, img, sigmax, row, T):
        if img.ndim != 3 and img.ndim != 2:
            raise ValueError('Image is not greycale or rgb')
        elif img.ndim == 3:
            'convert to greyscale if image is rgb'
            Gimg = rgb2grey(img)
        else:
            Gimg = np.copy(img)
        GausImg = self.gaussian(Gimg, sigmax)
        DoG = GausImg - GausImg
        U = Gimg + row*DoG
        T = np.ones((770, 1024))
        I_xDoG = T*U
        return self.LineCleanup(I_xDoG)

    def LineCleanup(self, img):
        horizontal_kernel = np.array([1, 0, - 1]).reshape((1, 3, 1))
        vertical_kernel = horizontal_kernel.T
        mean_averaging_filter = 1/9*np.full((3, 3), 1)
        E_horz = ndimage.convolve(img, horizontal_kernel, mode='nearest')
        E_vert = ndimage.convolve(img, vertical_kernel, mode='nearest')
        C = E_horz > 0 and E_vert > 0
        B = ndimage.convolve(img, mean_averaging_filter, mode='nearest')
        I_filtered = np.copy(img)
        I_filtered[C] = B[C]
        return I_filtered

    def gaussian(self, img, sigma):
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

        if (N % 2) == 0:
            N += 1

        vertical_kernel = np.ones((N, 1), dtype=np.float32)
        for i in range(0, N):
            vertical_kernel[i, 0] = (1/math.sqrt(2*math.pi*sigma**2)) * \
                math.exp(-(i-math.floor(0.5*N))**2/(2*sigma**2))
        horizontal_kernel = vertical_kernel.T
        vertical_kernel = vertical_kernel
        mask_sum = (vertical_kernel*horizontal_kernel).sum()
        vertical_conv = ndimage.convolve(img, vertical_kernel, mode='nearest')
        horizontal_kernel = ndimage.convolve(
            vertical_conv, horizontal_kernel, mode='nearest')
        gaus = (1/mask_sum)*horizontal_kernel
        return gaus
