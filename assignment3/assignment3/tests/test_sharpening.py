import pathlib

import numpy as np
import pytest
from skimage.io import imread, imsave

from assignment import sharpening


def test_q2a_apply_unsharp_mask(tmp_path):
    img = imread(pathlib.Path() / 'samples' / 'stones.jpg', as_gray=True)
    out = sharpening.unsharp_masking(img, 1.0, 2.0)
    assert out.max() == 1
    assert out.min() == 0
    imsave(tmp_path / 'stones-unsharpen.png', out)


def test_q2a_unsharp_mask_rejects_invalid_gain():
    with pytest.raises(ValueError):
        sharpening.unsharp_masking(np.zeros((3, 3)), -1.0, 2.0)


def test_q2b_apply_laplacian_sharpening(tmp_path):
    img = imread(pathlib.Path() / 'samples' / 'pavement.jpg', as_gray=True)
    out = sharpening.laplacian(img, 0.5)
    assert out.max() == 1
    assert out.min() == 0
    imsave(tmp_path / 'pavement-laplacian.png', out)


def test_q2a_laplacian_sharpening_rejects_invalid_gain():
    with pytest.raises(ValueError):
        sharpening.laplacian(np.zeros((3, 3)), -1.0)
