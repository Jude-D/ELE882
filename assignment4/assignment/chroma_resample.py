import colour_space
from skimage import io
import numpy as np
import sys


def chroma_resample(input_image, output_image, sampling_factor):
    image = io.imread(input_image)/255
    print(np.max(image), np.min(image))
    converter = colour_space.YCbCrColourSpace(sampling_factor)
    Y, CbCr = converter.to_ycbcr(image)
    print(Y.shape, CbCr.shape)
    print(sampling_factor)
    rgb = np.clip(converter.to_rgb(Y, CbCr), 0, 1)
    io.imsave(output_image, rgb)


if __name__ == '__main__':
    chroma_resample(sys.argv[1], sys.argv[2], float(sys.argv[3]))
