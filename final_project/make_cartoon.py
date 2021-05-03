from skimage import io
import numpy as np
import sys
from cartoon_effect.cartoon_effect import CartoonEffect


def makecartoon(input_image, sigmax, row, T, sigmab, N, output_image):
    image = io.imread(input_image)
    convert = CartoonEffect()
    N = int(N)
    convert.apply(image, sigmax, row, T, sigmab, N)
    io.imsave(output_image, apply)


if __name__ == '__main__':
    makecartoon(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]),
                sys.argv[4], float(sys.argv[5]), float(sys.argv[6]), sys.argv[7])
