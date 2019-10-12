#!/usr/bin/env python
"""neural_network_model.py:
Implementing deep neural network in Python

Author     : Shion Fujimori
Date       : April 6th 2019
Affiliation: University of Toronto, Computer Science Specialist
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from Deep_Neural_Network.forward_propagation import \
    initialize_parameters, forward_propagation, compute_cost
from Deep_Neural_Network.backward_propagation import \
    backward_propagation, update_parameters


def neural_network_model(X: np.array, Y: np.array, layer_dims: List,
                         learning_rate: float, num_iterations: int,
                         print_cost: bool) -> Tuple:
    """Implementing deep neural network

    :param X: Input data, shape (input size, number of examples)
    :param Y: True "label" vector, shape (1, number of examples)
    :param layer_dims: List containing the input size and each layer size
    :param learning_rate: Learning rate of the gradient descent update rule
    :param num_iterations: Number of iterations of the optimization loop
    :param print_cost: If True, it prints the cost every 100 steps
    :return: parameters learnt by the model
    """
    costs = []  # keep track of cost
    iterations = []  # keep track of number of iterations

    parameters = initialize_parameters(layer_dims)

    for i in range(1, num_iterations+1):
        # forward propagation: [LINEAR->ReLU]*(L-1) -> LINEAR -> SIGMOID
        AL, caches = forward_propagation(X, parameters)

        # Compute cost
        cost = compute_cost(AL, Y)

        # backward propagation: dAL -> SIGMOID -> LINEAR -> [ReLU->LINEAR]*(L-1)
        grads = backward_propagation(AL, Y, caches)

        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # print the cost every 100 training iterations (if print_cost == True)
        if print_cost and (i % 100 == 0 or i == 1):
            print("Cost after iteration {} iterations: {}".format(i, cost))
            # calculate the accuracy of the model
            print("Accuracy: " + str(np.average((AL >= 0.5) == Y)*100) + " %")
        costs.append(cost)
        iterations.append(i)

    # plot the cost
    plt.plot(iterations, costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.title('Cost vs Iterations (Learning rate = {}, Layer_dims = {})'
              .format(str(learning_rate), str(layer_dims)))
    plt.show()

    return parameters, costs
