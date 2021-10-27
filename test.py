import numpy
import torch

from networks.fully_connected_neural_network import NeuralNetwork
from solvers.finite_difference import compute_finite_difference_solution

def test(network, test_set):
    expected_results = [
        compute_finite_difference_solution(data_point[0], data_point[1]).tolist() for data_point in test_set
    ]

    network_results = [
        network(torch.tensor(
            numpy.concatenate((data_point[0], data_point[1])), dtype=torch.float32)
        ).detach().numpy().tolist() for data_point in test_set
    ]

    return numpy.mean(
        [numpy.sum(numpy.abs(numpy.array(expected_out) - numpy.array(model_out)))
         for expected_out, model_out in zip(expected_results, network_results)]
    )
