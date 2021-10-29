import argparse
import numpy
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

from itertools import product
from itertools import chain
from test import test_fully_connected_neural_network, test_deep_operator_network
from networks.fully_connected_neural_network import NeuralNetwork
from networks.deep_operator_network import UnstackedDeepOperatorNetwork
from data.utils import read_data_set
from prettytable import PrettyTable

os.chdir(Path(__file__).absolute().parent)
with open("config/config.json") as config_file:
    config = json.load(config_file)


def build_neural_network(width, depth, learning_rate, momentum, weight_decay):
    input_layer = [2 * config["parameters"]["number_of_grid_points"]]
    hidden_layers = [width] * depth
    output_layer = [config["parameters"]["number_of_grid_points"]]
    return NeuralNetwork(input_layer, hidden_layers,
                         output_layer, learning_rate, momentum, weight_decay)


def build_unstacked_deep_operator_network(branch_width, branch_depth, branch_learning_rate, branch_momentum, branch_weight_decay,
                                          trunk_width, trunk_depth, trunk_learning_rate, trunk_momentum, trunk_weight_decay):
    branch_input_layer = [2 * config["parameters"]["number_of_grid_points"]]
    branch_hidden_layers = [branch_width] * branch_depth
    branch_output_layer = [config["parameters"]["number_of_grid_points"]]
    trunk_input_layer = [1]
    trunk_hidden_layers = [trunk_width] * trunk_depth
    trunk_output_layer = [config["parameters"]["number_of_grid_points"]]
    return UnstackedDeepOperatorNetwork(
        branch_input_layer,
        branch_hidden_layers,
        branch_output_layer,
        branch_learning_rate,
        branch_momentum,
        branch_weight_decay,
        trunk_input_layer,
        trunk_hidden_layers,
        trunk_output_layer,
        trunk_learning_rate,
        trunk_momentum,
        trunk_weight_decay)


parser = argparse.ArgumentParser(description="TODO: Description")
parser.add_argument("Datafile", metavar="datafile",
                    type=str, help="File to read data from")

args = parser.parse_args()

print()

# Hyperparameter space for the grid search
# params = {
#    'width': list(range(10, 2560)),
#    'depth': [2, 3, 4, 5, 6],
#    'lr': [0.1, 0.01, 0.001, 0.0001]
# }

params_fully_connected_neural_networks = {
    'width': [64],
    'depth': [4],
    'learning_rate': [0.01]
}


params_unstacked_deep_operator_networks = {
    'branch_width': [64],
    'branch_depth': [4],
    'branch_learning_rate': [0.01],
    'trunk_width': [64],
    'trunk_depth': [4],
    'trunk_learning_rate': [0.01]
}

avg_test_errors = {}
error_table = PrettyTable()
error_table.field_names = ['Width', 'Depth', 'Learning rate', 'Error']


# Set up plot for training errors
fig, ax = plt.subplots()

for param_choice in list(product(*params_fully_connected_neural_networks.values())):
    print(
        'Training fully conected neural network with width={}, depth={} and learning_rate={}'.format(
            param_choice[0],
            param_choice[1],
            param_choice[2]))

    network = build_neural_network(
        param_choice[0], param_choice[1], param_choice[2], 0.9, 0.3)

    training_data = numpy.array_split(read_data_set(args.Datafile), 2)
    test_errors = []

    for i in range(len(training_data)):
        # Prepare data for cross validation
        test_set = training_data[i]
        training_set = chain(*(training_data[0:i] + training_data[i + 1:]))

        # Train model
        network.train_model(training_set)

        # Compute error for current test set
        test_errors.append(test_fully_connected_neural_network(network, test_set))

    # if param_choice[2] == params_fully_connected_neural_networks['learning_rate'][0]:
    #     ax.plot(list(range(network.number_of_performed_training_steps)),
    #             network.training_error_log)

    avg_test_error = numpy.average(test_errors)
    avg_test_errors[param_choice] = avg_test_error
    error_table.add_row([param_choice[0], param_choice[1],
                         param_choice[2], avg_test_error])

fig.savefig("plots/Training_Errors.png")

print()
print(error_table)
print()

# width_errors = []

# for param_choice, error in avg_test_errors.items():
#     if param_choice[1] == params['depth'][0]:
#        width_errors.append(error)

# fig, ax = plt.subplots()
# ax.plot(params['width'], width_errors)
# fig.savefig('plots/Width_Errors.png')

# param_choice_min_error = min(avg_test_errors, key=avg_test_errors.get)
# print('Parameter choice with minimal error: {}'.format(param_choice_min_error))


for param_choice in list(product(*params_unstacked_deep_operator_networks.values())):
    print("Training unstacked deep operator network with parameters")

    print("branch_width={}, branch_depth={} and branch_learning_rate={}".format(
        param_choice[0], param_choice[1], param_choice[2]))

    print("trunk_width={}, trunk_depth={} and trunk_learning_rate={}".format(
        param_choice[3], param_choice[4], param_choice[5]))

    network = build_unstacked_deep_operator_network(
        param_choice[0],
        param_choice[1],
        param_choice[2],
        0.9,
        0.3,
        param_choice[3],
        param_choice[4],
        param_choice[5],
        0.9,
        0.3)

    training_data = numpy.array_split(read_data_set(args.Datafile), 2)
    test_errors = []

    for i in range(len(training_data)):
        # Prepare data for cross validation
        test_set = training_data[i]
        training_set = chain(*(training_data[0:i] + training_data[i + 1:]))

        # Train model
        network.train_model(training_set)

        # Compute error for current test set
        test_errors.append(test_deep_operator_network(network, test_set))
