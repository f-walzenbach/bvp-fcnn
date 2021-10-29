import os
from pathlib import Path
import json
import numpy
import torch
import torch.nn as nn

from networks.fully_connected_neural_network import NeuralNetwork
from solvers.finite_difference import compute_finite_difference_solution

os.chdir(Path(__file__).absolute().parent)
with open("../config/config.json") as config_file:
    config = json.load(config_file)

class UnstackedDeepOperatorNetwork(nn.Module):
    def __init__(
            self,
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
            trunk_weight_decay,
        ):
        super(UnstackedDeepOperatorNetwork, self).__init__()

        # Dict of training_step -> loss
        self.training_error_log = {}
        self.current_training_step = 0

        # TODO: Insert assertion for branch output == trunk output
        self.branch = NeuralNetwork(branch_input_layer, branch_hidden_layers, branch_output_layer,
                                       branch_learning_rate, branch_momentum, branch_weight_decay)

        self.trunk = NeuralNetwork(trunk_input_layer, trunk_hidden_layers, trunk_output_layer,
                                       trunk_learning_rate, trunk_momentum, trunk_weight_decay)

        self.loss_fn = nn.MSELoss()

    def forward(self, u, y):
        return torch.matmul(self.branch(u), self.trunk(y))

    def backward(self, output, target):
        self.branch.optimizer.zero_grad()
        self.trunk.optimizer.zero_grad()

        loss = self.loss_fn(output, target)
        loss.backward()

        self.current_training_step += 1
        self.training_error_log[self.current_training_step] = loss.item()

        self.branch.optimizer.step()
        self.trunk.optimizer.step()

    def train_model(self, training_set):
        start = config["parameters"]["lower_bound"]
        stop = config["parameters"]["upper_bound"]
        num = config["parameters"]["number_of_grid_points"]
        domain = numpy.linspace(start, stop, num)

        for data_point in training_set:
            training_data = compute_finite_difference_solution(data_point[0], data_point[1])

            model_input = numpy.concatenate((data_point[0], data_point[1]))

            for grid_point, target in zip(domain, training_data):
                output = self(torch.tensor(model_input, dtype=torch.float32),
                              torch.tensor([grid_point], dtype=torch.float32))
                self.backward(output, torch.tensor(
                    target, dtype=torch.float32))

    def load(self, path):
        state_dict = torch.load(path)
        self.branch.load_state_dict(state_dict["branch"])
        self.trunk.load_state_dict(state_dict["trunk"])

    def save(self, path):
        torch.save({
            "branch": self.branch.state_dict(),
            "trunk": self.trunk.state_dict()
        }, path)
