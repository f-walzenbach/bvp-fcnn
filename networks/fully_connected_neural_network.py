import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from solvers.finite_difference import compute_finite_difference_solution

class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_layer,
        hidden_layers,
        output_layer,
        learning_rate,
        momentum,
        weight_decay
    ):
        super(NeuralNetwork, self).__init__()

        # Dict of training_step -> loss
        self.training_error_log = {}
        self.current_training_step = 0

        layers = numpy.concatenate((
            input_layer,
            hidden_layers,
            output_layer
        ))

        self.hidden = nn.ModuleList()

        for i in range(0, len(layers) - 1):
            self.hidden.append(
                nn.Linear(layers[i], layers[i + 1]))

        self.loss_fn = nn.MSELoss()

        self.optimizer = optim.SGD(
            self.parameters(),
            learning_rate,
            momentum,
            weight_decay
        )

    def forward(self, x):
        for hidden_layer in self.hidden:
            x = hidden_layer(x)
        return x

    def backward(self, output, target):
        self.zero_grad()
        self.optimizer.zero_grad()

        loss = self.loss_fn(target, output)
        loss.backward()
        self.optimizer.step()

        self.current_training_step += 1
        self.training_error_log[self.current_training_step] = loss.item()

    def train_model(self, training_set):
        for data_point in training_set:
            training_data = compute_finite_difference_solution(data_point[0], data_point[1])

            model_input = numpy.concatenate((data_point[0], data_point[1]))

            network_output = self(torch.tensor(
                model_input, dtype=torch.float32))
            expected_output = torch.tensor(training_data, dtype=torch.float32)

            self.backward(network_output, expected_output)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)
