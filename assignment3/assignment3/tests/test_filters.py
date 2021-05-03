import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_approx_equal
import pytest

from assignment import filters


def test_q1a_test_laplacian_filter():
    # Impulse response should be the Laplacian kernel
    img = np.zeros((3, 3))
    img[1, 1] = 1

    expected = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    out = filters.laplacian(img)
    assert_array_equal(out, expected)


def test_q1a_test_laplacian_rejects_ubyte_images():
    with pytest.raises(TypeError):
        filters.laplacian(np.zeros((3, 3), dtype=np.uint8))


def test_q1a_test_laplacian_rejects_colour_images():
    with pytest.raises(ValueError):
        filters.laplacian(np.zeros((3, 3, 3)))


def test_q1b_test_gaussian_filter():
    # Impulse response should be the Gaussian kernel (values are hand-computed).
    img = np.zeros((5, 5))
    img[2, 2] = 1

    expected = np.array([
        [4.78206e-05, 0.00135148, 0.00411664, 0.00135148, 4.78206e-05],
        [0.00135148, 0.038195, 0.116342, 0.038195, 0.00135148],
        [0.00411664, 0.116342, 0.354381, 0.116342, 0.00411664],
        [0.00135148, 0.038195, 0.116342, 0.038195, 0.00135148],
        [4.78206e-05, 0.00135148, 0.00411664, 0.00135148, 4.78206e-05]
    ])

    out = filters.gaussian(img, 0.67)
    assert_approx_equal(out.sum(), 1)
    assert_allclose(out, expected, atol=1e-6)


def test_q1b_test_gaussian_rejects_ubyte_images():
    with pytest.raises(TypeError):
        filters.gaussian(np.zeros((3, 3), dtype=np.uint8), 1)


def test_q1b_test_gaussian_rejects_colour_images():
    with pytest.raises(ValueError):
        filters.gaussian(np.zeros((3, 3, 3)), 1)


def test_q1b_test_gaussian_rejects_invalid_sigma():
    with pytest.raises(ValueError):
        filters.gaussian(np.zeros((3, 3)), 1e-9)

    with pytest.raises(ValueError):
        filters.gaussian(np.zeros_like((3, 3)), -1)


def test_q1c_test_moving_average_filter():
    # Impulse response should be the moving average kernel (all values 1/N^2).
    img = np.zeros((3, 3))
    img[1, 1] = 1

    out = filters.moving_average(img, 3)
    assert_allclose(out, 1/9)


def test_q1c_test_moving_average_rejects_ubyte_images():
    with pytest.raises(TypeError):
        filters.moving_average(np.zeros((3, 3), dtype=np.uint8), 3)


def test_q1c_test_moving_average_rejects_colour_images():
    with pytest.raises(ValueError):
        filters.moving_average(np.zeros((3, 3, 3)), 3)


def test_q1c_test_moving_average_rejects_invalid_width():
    with pytest.raises(ValueError):
        filters.moving_average(np.zeros((3, 3)), 0)

    with pytest.raises(ValueError):
        filters.moving_average(np.zeros((3, 3)), -1)
