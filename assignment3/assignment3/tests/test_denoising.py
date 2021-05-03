from skimage import data
from skimage.io import imsave
from skimage.util import img_as_ubyte

from assignment.noise import additive, impulsive, multiplicative

from noise_process import AdditiveNoise, ImpulsiveNoise, MultiplicativeNoise


def run_denoise(img, process, fn, folder):
    '''Helper for running the denoising tests.

    Parameters
    ----------
    img : numpy.ndarray
        image being processed
    process : NoiseProcess
        a noise process instance
    fn : function
        the denoising function
    folder : path-like object
        path to where the output images should be saved
    '''
    imsave(folder / 'original.png', img)
    noisy, denoised = process.denoise(fn, img)
    imsave(folder / 'noisy.png', img_as_ubyte(noisy))
    imsave(folder / 'denoised.png', img_as_ubyte(denoised))


def test_q5a_additive_denoise(tmp_path):
    img = data.chelsea()
    run_denoise(img, AdditiveNoise(0.05), additive.denoise, tmp_path)


def test_q5b_multiplicative_denoise(tmp_path):
    img = data.chelsea()
    run_denoise(img, MultiplicativeNoise(0.15), multiplicative.denoise, tmp_path)


def test_q5c_impulsive_denoise(tmp_path):
    img = data.chelsea()
    run_denoise(img, ImpulsiveNoise(0.01), impulsive.denoise, tmp_path)
