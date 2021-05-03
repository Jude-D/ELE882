import numpy as np


def imread(filename):
    '''Load a NetPBM image from a file.

    Parameters
    ----------
    filename : str
        image file name

    Returns
    -------
    numpy.ndarray
        a numpy array with the loaded image

    Raises
    ------
    ValueError
        if the image format is unknown or invalid
    '''
    # Read in the file
    with open(filename, 'rt') as f:
        contents = f.read()

    # Split the file contents into it's constituents tokens
    tokens = contents.split()
    if tokens[0] != 'P2' and tokens[0] != 'P3':
        raise ValueError(f'Unknown format {tokens[0]}')

    # Get the image dimensions
    width = int(tokens[1])
    height = int(tokens[2])
    maxval = int(tokens[3])

    # Convert all if the string tokens into integers
    values = [int(token) for token in tokens[4:]]
    if maxval != 255:
        raise ValueError('Can only support 8-bit images.')

    # Create the numpy array, reshaping it so that it's no longer a linear array
    if tokens[0] == 'P2':
        image = np.array(values, dtype=np.uint8)
        return np.reshape(image, (height, width))
    elif tokens[0] == 'P3':
        image = np.array(values, dtype=np.uint8)
        return np.reshape(image, (height, width, 3))
        # print(values)
        # print(image)
        # print(image.shape)


def imwrite(filename, image):
    '''Save a NetPBM image to a file.

    Parameters
    ----------
    filename : str
        image file name
    image : numpy.ndarray
        image being saved
    '''
    # Extract the image dimensions and check that it's a 8bc
    if image.ndim == 2:
        height, width = image.shape
    elif image.ndim == 3:
        width, height, layer = image.shape
    if image.dtype != np.uint8:
        raise ValueError('Can only support 8-bit images.')

    # Convert the image values to strings
    values = image.astype(str)

    def dim2(dim2arr, rowjoin):
        rows = [' '.join(row) for row in dim2arr]
        data = rowjoin.join(rows)
        return data
    # Contruct the file contents
    if image.ndim == 2:
        header = ' \n'.join(['P2', str(width), str(height), '255'])
        data = dim2(values, ' \n')
    elif image.ndim == 3:
        header = ' \n'.join(['P3', str(width), str(height), '255'])
        data = ' \n'.join(dim2(list_layer, ' ') for list_layer in values)

    # Write it to a file
    with open(filename, 'wt') as f:
        f.write(header)
        f.write('\n')
        f.write(data)

    # raise NotImplementedError('Implement this function/method.')
