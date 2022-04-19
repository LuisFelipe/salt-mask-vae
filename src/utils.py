# -*- coding: utf-8 -*-
"""
module utils.py
--------------------
A set of utility functions to be used in the model implementations.
"""
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from scipy import ndimage


def mkdir_if_not_exists(path):
    """
    make directory if it does not exists.
    :param path: dir path.
    :return: True if the path was created. False otherwise.
    """
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False


def save_figure_from_array(path, array):
    """
    Saves the array as image file.
    :param path: file path.
    :param array: figure numpy array.
    :return: True if success.
    """
    array = array.astype('uint8') * 255
    img = Image.fromarray(array, mode='L')
    img.save(fp=path)
    img.close()


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def mask_post_process(x):
    """Post processing of random generated mask.

    Remove white and black pixels noise.
    :param x: the generated mask.
    :type x: np.ndarray
    :returns: the generated mask with noise removed.
    :rtype: np.ndarray
    """
    # Remove small white regions 
    # x = ndimage.binary_opening(x, iterations=2)
    # Remove small black hole
    # x = ndimage.binary_closing(x, iterations=2)
    x = ndimage.binary_dilation(x, iterations=2)
    return x
