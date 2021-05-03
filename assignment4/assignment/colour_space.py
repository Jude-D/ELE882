import numpy as np
from skimage.transform import rescale, resize
# from skimage.util import img_as_float


class YCbCrColourSpace:
    '''Convert an image between the YCbCr and RGB colour spaces.

    The class implements the JPEG varient of YCbCr with support for optional
    chroma subsampling.  The subsampling is represented as a positive integer,
    so a subsampling factor of '2' means the dimensions of the chroma channels
    are half of the luma channel.

    Attributes
    ----------
    sampling : int
        subsampling factor
    '''

    def __init__(self, sampling=1):
        '''Initialize the YCbCr to RGB converter.

        Parameters
        ----------
        sampling : int, optional
            the YCbCr chroma subsampling factor; defaults to '1' or no
            subsampling

        Raises
        ------
        ValueError
            if the sampling factor is less than '1'
        '''
        if sampling < 1:
            raise ValueError('Sampling factor must be larger than "1".')
        self.sampling = sampling

    def to_ycbcr(self, img):
        '''Convert the input RGB image into YCbCr.

        Parameters
        ----------
        img : numpy.ndarray
            input RGB image

        Returns
        -------
        Y : numpy.ndarray
            the luma channel, the same width and height as the input image
        CbCr : numpy.ndarray
            a two-channel image containing the (possibly) subsampled chroma
            channels

        Raises
        ------
        ValueError
            if the image isn't a 3-channel colour image
        '''
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError('Image is not a 3-channel colour image')
        img = img.astype(float)
        forward_transform = np.array([
            [0.299, 0.587, 0.114],
            [-0.1687, -0.3313, 0.5],
            [0.5, -0.4187, -0.0813]])
        YCbCr = img.dot(forward_transform.T)
        Y = YCbCr[:, :, 0]
        CbCr = YCbCr[:, :, 1:]
        CbCr = np.stack([self._downsample(CbCr[:, :, 0]),
                        self._downsample(CbCr[:, :, 1])], axis=2)
        print(str(np.min(CbCr)) + '  ' + str(np.max(CbCr)))
        return np.clip(Y, -1, 1), np.clip(CbCr, -0.5, 0.5)

    def to_rgb(self, Y, CbCr):
        '''Convert the input YCbCr image into RGB.

        The chroma channels are first upsampled to have the same width and
        height as the luma channel before the YCbCr to RGB conversion.  They may
        not be larger than the luma channel.

        Parameters
        ----------
        Y : numpy.ndarray
            luma channel
        uv : numpy.ndarray
            chroma channels; possibly subsampled

        Returns
        -------
        numpy.ndarray
            reconstructed RGB image

        Raises
        ------
        ValueError
            if ``Y`` isn't a single-channel image or if ``uv`` isn't a
            two-channel image and smalelr than ``Y``
        '''
        if Y.ndim != 2:
            raise ValueError("Y isn't single-channel image")
        elif CbCr.ndim != 3 or CbCr.shape[2] != 2 or CbCr.shape[0] > Y.shape[0] \
                or CbCr.shape[1] > Y.shape[1]:
            raise ValueError("uv isn't a two-channel image and smaller than Y")
        CbCr = np.stack([self._upsample(CbCr[:, :, 0], (Y.shape[0], Y.shape[1])),
                        self._upsample(CbCr[:, :, 1], (Y.shape[0], Y.shape[1]))], axis=2)
        Y = np.expand_dims(Y.astype(float), axis=2)
        CbCr = CbCr.astype(float)
        reverse_transform = np.array(
            [[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
        print(str(CbCr.shape), str(Y.shape))
        YCbCr = np.concatenate([Y, CbCr], axis=2)
        rgb = np.clip(YCbCr.dot(reverse_transform.T), -1, 1)
        return rgb

    def _downsample(self, c):
        '''Downsample a single-channel image.

        Parameters
        ----------
        c : numpy.ndarray
            input image
        '''
        return rescale(c, 1 / self.sampling, mode='edge', anti_aliasing=True)

    def _upsample(self, c, outsz):
        '''Upsample a single-channel image to match a specified output size.

        Parameters
        ----------
        c : numpy.ndarray
            input image
        outsz : tuple of ``(height, width)``
            the expected output size
        '''
        return resize(c, outsz, mode='edge', anti_aliasing=True)
