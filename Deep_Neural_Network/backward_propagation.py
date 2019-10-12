#!/usr/bin/env python
"""backward_propagation.py:
Implementing backward propagation algorithm in Python

Author     : Shion Fujimori
Date       : April 6th 2019
Affiliation: University of Toronto, Computer Science Specialist
"""

import numpy as np
from typing import List, Dict, Tuple
from Deep_Neural_Network.activation_functions \
    import sigmoid_backward, relu_backward


def linear_backward(dZ: np.array, cache: Tuple) -> Tuple:
    """Implement the linear part of a layer's backward propagation

    dW[l] = (dZ[l]A[l-1].T)/m
    db[l] = (np.sum(dZ, axis=1, keepdims=True))/m
    dA[l-1] = W[l].T dZ[l]

    :param dZ: Gradient of the cost with respect to the linear output
    :param cache: Tuple of values (A_prev, W, b)
    :return: Tuple containing dA_prev, dW, db

    dA_prev: gradient of the cost with respect to the activation
             of the previous layer
    dW:      gradient of the cost with respect to W
    db:      gradient of the cost with respect to b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]  # number of examples

    dW = (np.dot(dZ, A_prev.T))/m
    db = (np.sum(dZ, axis=1, keepdims=True))/m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA: np.array,
                               cache: Tuple, activation: str) -> Tuple:
    """Implement the backward propagation for the LINEAR->ACTIVATION layer

    dZ[l] = dA[l]*g'(Z[l]) where g is the activation function
    dW[l] = (dZ[l]A[l-1].T)/m
    db[l] = (np.sum(dZ, axis=1, keepdims=True))/m
    dA[l-1] = W[l].T dZ[l]

    :param dA: gradient of the cost with respect to the activation
    :param cache: Tuple of values (linear_cache, activation_cache)
    :param activation: The activation to be used in this layer,
                       "sigmoid" or "relu"
    :return: Tuple containing dA_prev, dW, db

    dA_prev: gradient of the cost with respect to the activation
             of the previous layer
    dW:      gradient of the cost with respect to W
    db:      gradient of the cost with respect to b
    """
    linear_cache, activation_cache = cache
    dZ = np.array

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)

    return linear_backward(dZ, linear_cache)


def backward_propagation(AL: np.array, Y: np.array, caches: List) -> Dict:
    """Implement full backward propagation

    dAL -> SIGMOID -> LINEAR -> [ReLU->LINEAR] * (L-1) computation


    :param AL: Last post-activation value
    :param Y:  True "label" vector, shape (1, number of examples)
    :param caches: list of caches containing every
                   cache of linear_activation_forward()
    :return: Dictionary containing gradients: dA, dW, db
    """
    grads = {}
    L = len(caches)  # number of layers

    # Initialize the back propagation
    dAL = -np.divide(Y, AL) + np.divide(1-Y, 1-AL)

    # Implement SIGMOID -> LINEAR
    current_cache = caches[L-1]
    grads['dA'+str(L-1)], grads['dW'+str(L)], grads['db'+str(L)] = \
        linear_activation_backward(dAL, current_cache, 'sigmoid')

    # Implement [ReLU->LINEAR] * (L-1)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dA' + str(l)], grads['dW'+str(l+1)], grads['db'+str(l+1)] = \
            linear_activation_backward(grads['dA'+str(l+1)],
                                       current_cache, 'relu')
    return grads


def update_parameters(parameters: Dict,
                      grads: Dict, learning_rate: float) -> Dict:
    """Update parameters using gradient descent

    W[l] -= learning_rate * dW[l]
    b[l] -= learning_rate * db[l]

    :param parameters: Dictionary containing the parameters
                       "W1", "b1", ... , "WL", "bL"
    :param grads: Dictionary containing gradients: dA, dW, db
    :param learning_rate: A hyper-parameter that controls how much we are
                          adjusting the weights of the network with respect
                          the loss gradient.
    :return: Dictionary containing the updated parameters
             "W1", "b1", ... , "WL", "bL"
    """
    L = len(parameters)//2  # number of layers

    for l in range(1, L+1):
        parameters['W'+str(l)] -= learning_rate * grads['dW'+str(l)]
        parameters['b'+str(l)] -= learning_rate * grads['db'+str(l)]

    return parameters
