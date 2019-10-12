#!/usr/bin/env python
"""experiment.py:
Implementing multiple helper functions for the experiment

Author     : Shion Fujimori
Date       : September 20th 2019
Affiliation: University of Toronto, Computer Science Specialist
"""
from Deep_Neural_Network.neural_network_model import neural_network_model as nn
from typing import Dict, List
import numpy as np


def normalize(input_data: np.array):
    for row in range(input_data.shape[0]):
        average = np.average(input_data[row])
        ranges = np.max(input_data[row]) - np.min(input_data[row])
        input_data[row] = (input_data[row] - average) / ranges
    return input_data


def layer_dims_experiment(layer_dims: List, hyper_parameters: Dict) -> Dict:
    """Train the neural network with various layer dimensions to determine
    which dimensions works the best.

    :param layer_dims: List of layer dimension
    :param hyper_parameters: hyper parameters
    :return: dictionary {layer_dims: cost}
    """
    result = {}

    input_data = hyper_parameters["input_data"]
    output_data = hyper_parameters["output_data"]
    learning_rate = hyper_parameters["learning_rate"]
    iterations = hyper_parameters["iterations"]
    print_cost = hyper_parameters["print_cost"]

    for layer in layer_dims:
        _, costs = nn(input_data, output_data, layer, learning_rate, iterations, print_cost)
        result[str(layer)] = costs[-1]
    return result
