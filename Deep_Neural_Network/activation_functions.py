#!/usr/bin/env python
"""activation_functions.py:
Implementing activation functions algorithm (sigmoid, ReLU)

Author     : Shion Fujimori
Date       : April 6th 2019
Affiliation: University of Toronto, Computer Science Specialist
"""

import numpy as np
from typing import Tuple


def sigmoid(Z: np.array) -> Tuple:
    """Implement the sigmoid activation function

    :param Z: Output of the linear part,
              shape (size of current layer, number of examples)
    :return: Tuple containing A and cache

    A:     the output of the activation function (i.e. post-activation value)
    cache: "Z", stored for computing the backward propagation efficiently
    """
    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z: np.array) -> Tuple:
    """Implement the ReLU activation function

    ReLU: Rectified Linear Unit

    :param Z: Output of the linear part,
              shape (size of current layer, number of examples)
    :return: Tuple containing A and cache

    A:     the output of the activation function (i.e. post-activation value)
    cache: "Z", stored for computing the backward propagation efficiently
    """
    A = np.maximum(0, Z)
    cache = Z

    return A, cache


def sigmoid_backward(dA: np.array, Z: np.array) -> np.array:
    """Implement the backward propagation for a single sigmoid unit

    :param dA: post-activation gradient
    :param Z:  Output of the linear part, cache "Z"
    :return:   dZ, gradient of the cost with respect to Z
    """
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    return dZ


def relu_backward(dA: np.array, Z: np.array) -> np.array:
    """Implement the backward propagation for a single ReLU unit

    :param dA: post-activation gradient
    :param Z:  Output of the linear part, cache "Z"
    :return:   dZ, gradient of the cost with respect to Z
    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ
