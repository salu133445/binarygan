"""Utilities for creating image grids from a batch of images.
"""
import numpy as np
import imageio

def get_image_grid(images, shape, grid_width=0, grid_color=0,
                   frame=False):
    """
    Merge the input images and return a merged grid image.

    Arguments
    ---------
    images : np.array, ndim=3 or 4
        The image array. Shape is (num_image, height, width, (num_channel)).
    shape : list or tuple of int
        Shape of the image grid. (height, width)
    grid_width : int
        Width of the grid lines. Default to 0.
    grid_color : int
        Color of the grid lines. Available values are 0 (black) to
        255 (white). Default to 255.
    frame : bool
        True to add frame. Default to False.

    Returns
    -------
    merged : np.array, ndim=3
        The merged grid image.
    """
    if images.ndim == 3:
        reshaped = images.reshape(shape[0], shape[1], images.shape[1],
                                  images.shape[2], 1)
    elif images.ndim == 4:
        reshaped = images.reshape(shape[0], shape[1], images.shape[1],
                                  images.shape[2], images.shape[3])
    else:
        raise ValueError("Number of dimension for `images` must be 3 or 4")
    pad_width = ((0, 0), (0, 0), (grid_width, 0), (grid_width, 0), (0, 0))
    padded = np.pad(reshaped, pad_width, 'constant', constant_values=grid_color)
    transposed = padded.transpose(0, 2, 1, 3, 4)
    merged = transposed.reshape(shape[0] * (images.shape[1] + grid_width),
                                shape[1] * (images.shape[2] + grid_width),
                                reshaped.shape[4])
    if frame:
        return np.pad(merged, ((0, grid_width), (0, grid_width), (0, 0)),
                      'constant', constant_values=grid_color)
    return merged[grid_width:, grid_width:]

def save_image(filepath, phrases, shape, inverted=False, grid_width=1,
               grid_color=255, frame=False):
    """
    Save a batch of phrases to a single image grid.

    Arguments
    ---------
    filepath : str
        Path to save the image grid.
    phrases : np.array, ndim=5
        The phrase array. Shape is (num_phrase, num_bar, num_time_step,
        num_pitch, num_track).
    shape : list or tuple of int
        Shape of the image grid. (height, width)
    inverted : bool
        True to invert the colors. Default to False.
    grid_width : int
        Width of the grid lines. Default to 2.
    grid_color : int
        Color of the grid lines. Available values are 0 (black) to
        255 (white). Default to 255.
    frame : bool
        True to add frame. Default to False.
    """
    if phrases.dtype == np.bool_:
        if inverted:
            phrases = np.logical_not(phrases)
        clipped = (phrases * 255).astype(np.uint8)
    else:
        if inverted:
            phrases = 1. - phrases
        clipped = (phrases * 255.).clip(0, 255).astype(np.uint8)
    merged = get_image_grid(clipped, shape, grid_width, grid_color, frame)
    imageio.imwrite(filepath, merged)
