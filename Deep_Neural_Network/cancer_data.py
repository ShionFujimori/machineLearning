#!/usr/bin/env python
"""cancer_data.py:
Applying neural network model to the breast cancer dataset

Author     : Shion Fujimori
Date       : September 20th 2019
Affiliation: University of Toronto, Computer Science Specialist
"""

from sklearn.datasets import load_breast_cancer
from Deep_Neural_Network.experiment import layer_dims_experiment, normalize

cancer_data = load_breast_cancer()

# input_data / output_data
input_data = normalize(cancer_data.data.T)
output_data = cancer_data.target.reshape(1, -1)

# learning_rate / num_iterations / print_cost
learning_rate = 0.5
iterations = 1000
print_cost = True

# all hyper parameters (except layer_dims)
hyper_parameters = {"input_data": input_data,
                    "output_data": output_data,
                    "learning_rate": learning_rate,
                    "iterations": iterations,
                    "print_cost": print_cost}

# layer_dims
layer1 = [30, 70, 10, 5, 1]
layer2 = [30, 70, 15, 5, 1]
layer3 = [30, 70, 10, 3, 1]
layers = [layer1, layer2, layer3]

print(layer_dims_experiment(layers, hyper_parameters))
