import numpy
import torch
import json

from networks.fully_connected_neural_network import NeuralNetwork
from solvers.finite_difference import compute_finite_difference_solution

with open("config/config.json", "r") as config_file:
    config = json.load(config_file)


def test_fully_connected_neural_network(network, test_set):
    expected_results = [
        compute_finite_difference_solution(
            data_point[0],
            data_point[1]).tolist() for data_point in test_set]

    network_results = [
        network(torch.tensor(
            numpy.concatenate((data_point[0], data_point[1])), dtype=torch.float32)
        ).detach().numpy().tolist() for data_point in test_set
    ]

    return numpy.mean(
        [numpy.sum(numpy.abs(numpy.array(expected_out) - numpy.array(model_out)))
         for expected_out, model_out in zip(expected_results, network_results)]
    )


def test_deep_operator_network(network, test_set):
    start = config["parameters"]["lower_bound"]
    stop = config["parameters"]["upper_bound"]
    num = config["parameters"]["number_of_grid_points"]
    domain = numpy.linspace(start, stop, num)

    expected_results = [
        compute_finite_difference_solution(
            data_point[0],
            data_point[1]).tolist() for data_point in test_set]

    network_results = []
    for data_point in test_set:
        network_results.append(
            [network(
                torch.tensor(
                    numpy.concatenate(
                        (data_point[0],
                         data_point[1],
                         [i])),
                    dtype=torch.float32)).detach().numpy() for i in domain])
