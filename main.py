import argparse
import numpy
import json
import os

import matplotlib.pyplot as plt

from itertools import product
from itertools import chain
from test import test
from networks.fully_connected_neural_network import NeuralNetwork
from data.utils import read_data_set

with open("config/config.json") as config_file:
    config = json.load(config_file)


def build_neural_network(width, depth, learning_rate, momentum, weight_decay):
    input_layer = [2 * config["parameters"]["number_of_grid_points"]]
    hidden_layers = [width] * depth
    output_layer = [config["parameters"]["number_of_grid_points"]]
    return NeuralNetwork(input_layer, hidden_layers,
                         output_layer, learning_rate, momentum, weight_decay)


parser = argparse.ArgumentParser(description="TODO: Description")
parser.add_argument("Datafile", metavar="datafile",
                    type=str, help="File to read data from")

args = parser.parse_args()

# Hyperparameter space for the grid search
params = {
    'width': list(range(10, 2560)),
    'depth': [2, 3, 4, 5, 6],
    'lr': [0.1, 0.01, 0.001, 0.0001]
}

avg_test_errors = {}

for param_choice in list(product(*params.values())):
    print('Training model with width={}, depth={} and learning_rate={}'.format(
        param_choice[0], param_choice[1], param_choice[2]))

    network=build_neural_network(
        param_choice[0], param_choice[1], param_choice[2], 0.9, 0.3)

    training_data=numpy.array_split(read_data_set(args.Datafile), 10)
    test_errors=[]

    for i in range(len(training_data)):
        # Prepare data for cross validation
        test_set=training_data[i]
        training_set=chain(*(training_data[0:i] + training_data[i+1:]))

        # Train model
        network.train_model(training_set)

        # Compute error for current test set
        test_errors.append(test(network, test_set))

    avg_test_errors[param_choice] = numpy.average(test_errors)

param_choice_min_error = min(avg_test_errors, key=avg_test_errors.get)
print('Parameter choice with minimal error: {}'.format(param_choice_min_error))
