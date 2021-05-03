import numpy as np
from skimage import io, exposure
import sys, argparse


def clahe_image(img, clip, wsize):
    return exposure.equalize_adapthist(img, wsize, clip)
    
def convert_image(input_image, output_image, clip, wsize):
    image = (io.imread(input_image, as_gray=True)*255).astype(np.uint8)
    output = (clahe_image(image, clip, wsize)*255).astype(np.uint8)
    io.imsave(output_image, output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--clip", type=float, default=0.01)
    parser.add_argument("--wsize", type=int, default=None)
    args = parser.parse_args()
    convert_image(args.input, args.output, args.clip, args.wsize)
