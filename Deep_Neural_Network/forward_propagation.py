#!/usr/bin/env python
"""forward_propagation.py:
Implementing forward propagation algorithm in Python

Author     : Shion Fujimori
Date       : April 6th 2019
Affiliation: University of Toronto, Computer Science Specialist
"""

import numpy as np
from typing import List, Dict, Tuple
from Deep_Neural_Network.activation_functions import sigmoid, relu


def initialize_parameters(layer_dims: List) -> Dict:
    """Create and initialize the parameters of the deep neural network

    :param layer_dims: List containing the dimensions of
           each layer in the network
    :return: Dictionary containing the parameters "W1", "b1", ... , "WL", "bL"

    Wl: weight matrix of shape (layer_dims[l] layer_dims[l-1])
    b1: bias vector of shape (layer_dims[l], 1)
    """
    parameters = {}
    L_all = len(layer_dims)  # number of layers in the network + 1

    for l in range(1, L_all):
        parameters['W'+str(l)] = \
            np.random.randn(layer_dims[l], layer_dims[l-1])*0.1
        parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_forward(A: np.array, W: np.array, b: np.array) -> Tuple:
    """Implement the linear part of a layer's forward propagation

    Z[l] = W[l]A[l-1]+b[l]

    :param A: Activations from previous layer,
              shape (size of previous layer, number of examples)
    :param W: Weights matrix,
              shape (size of current layer, size of previous layer)
    :param b: Bias vector, shape (size of current layer, 1)
    :return: Tuple containing Z and cache

    Z:     the input of the activation function (i.e. pre-activation parameter)
    cache: tuple containing "A", "W", "b";
           stored for computing the backward propagation efficiently
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev: np.array, W: np.array,
                              b: np.array, activation: str) -> Tuple:
    """Implement the forward propagation for the LINEAR->ACTIVATION layer

    Z[l] = W[l]A_prev[l-1]+b[l]
    A[l] = relu(Z[l]) or sigmoid(Z[l])

    :param A_prev: Activations from previous layer,
                   shape (size of previous layer, number of examples)
    :param W: Weights matrix,
              shape (size of current layer, size of previous layer)
    :param b: Bias vector, shape (size of current layer, 1)
    :param activation: The activation to be used in this layer,
                       "sigmoid" or "relu"
    :return: Tuple containing A and cache

    A:     the output of the activation function (i.e. post-activation value)
    cache: tuple containing "linear_cache" ("A_prev", "W", "b")
           and "activation_cache" ("Z")
           stored for computing the backward propagation efficiently
    """
    Z, linear_cache = linear_forward(A_prev, W, b)

    A = np.array
    activation_cache = np.array

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def forward_propagation(X: np.array, parameters: Dict) -> Tuple:
    """Implement full forward propagation

    [LINEAR->ReLU]*(L-1)->LINEAR->SIGMOID computation

    :param X: Input data, shape (input size, number of examples)
    :param parameters: Dictionary containing the parameters
                       "W1", "b1", ... , "WL", "bL"
    :return: Tuple containing AL and caches

    AL:     last post-activation value
    caches: list of caches containing every cache of linear_activation_forward()
    """
    caches = []
    A = X
    L = len(parameters) // 2

    # Implement [LINEAR -> ReLU]*(L-1)
    for l in range(1, L):
        A_prev = A
        W = parameters['W'+str(l)]
        b = parameters['b'+str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, 'relu')
        caches.append(cache)

    # Implement LINEAR -> SIGMOID
    WL = parameters['W'+str(L)]
    bL = parameters['b'+str(L)]
    AL, cache = linear_activation_forward(A, WL, bL, 'sigmoid')
    caches.append(cache)

    return AL, caches


def compute_cost(AL: np.array, Y: np.array) -> float:
    """Implement the cross-entropy error function

    :param AL: last post-activation value
    :param Y:  true "label" vector, shape (1, number of examples)
    :return: cross-entropy cost
    """
    m = Y.shape[1]

    cost = (-1/m)*np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
    cost = float(np.squeeze(cost))

    return cost

