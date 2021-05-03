import sys

from oo import rgb2grey
from mm import imread, imwrite


def convert_image(rgb_image, grey_image):
    image = imread(rgb_image)
    greyscale = rgb2grey(image)
    imwrite(grey_image, greyscale)


if __name__ == '__main__':
    convert_image(sys.argv[1], sys.argv[2])
