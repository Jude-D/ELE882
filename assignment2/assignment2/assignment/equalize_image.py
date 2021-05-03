import numpy as np
import point_operators
import analysis
from skimage import io
import sys


def equalize_lut(hist):
    cdf = np.cumsum(hist, dtype=float)
    return (cdf / cdf[255] * 255.0).astype(np.uint8)


def equalize_image(image):
    output_hist = analysis.histogram(image)
    return point_operators.apply_lut(image, equalize_lut(output_hist))


def convert_image(input_image, output_image):
    image = (io.imread(input_image, as_gray=True)*255).astype(np.uint8)
    output = equalize_image(image)
    io.imsave(output_image, output)


if __name__ == '__main__':
    convert_image(sys.argv[1], sys.argv[2])
