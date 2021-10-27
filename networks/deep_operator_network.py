import numpy
import torch
import torch.nn as nn
from solvers.finite_difference import compute_finite_difference_solution


class Branch(nn.Module):
    def __init__(self, input_size):
        super(Branch, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
        )

    def forward(self, u):
        b = self.model(u)
        return b


class Trunk(nn.Module):
    def __init__(self):
        super(Trunk, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 128),
            nn.Sigmoid()
        )

    def forward(self, y):
        t = self.model(y)
        return t


class UnstackedDeepOperatorNetwork(nn.Module):
    def __init__(self, input_size):
        super(UnstackedDeepOperatorNetwork, self).__init__()

        self.label = "Deep_Operator_Network"

        self.branch = Branch(input_size)
        self.trunk = Trunk()

        self.branch_optimizer = torch.optim.SGD(
            self.branch.parameters(), lr=0.0001)

        self.trunk_optimizer = torch.optim.SGD(
            self.trunk.parameters(), lr=0.0001)

        self.loss_fn = nn.MSELoss()

    def forward(self, u, y):
        return torch.matmul(self.branch(u), self.trunk(y))

    def backward(self, output, target):
        self.branch_optimizer.zero_grad()
        self.trunk_optimizer.zero_grad()

        loss = self.loss_fn(output, target)
        loss.backward()

        self.branch_optimizer.step()
        self.trunk_optimizer.step()

    def train_model(self, training_set):
        print("Training Deep Operator Network")
        index = 1
        for data_point in training_set:
            print("Current training set index: {}".format(index))
            index += 1
            training_data = compute_finite_difference_solution(
                parameters.discretized_domain,
                parameters.LOWER_BOUNDARY_VALUE,
                parameters.UPPER_BOUNDARY_VALUE,
                data_point[0],
                data_point[1]
            )

            model_input = numpy.concatenate((data_point[0], data_point[1]))

            #for _ in range(0, parameters.NUMBER_OF_TRAINING_STEPS):
            for _ in range(0, 100):
                for grid_point, target in zip(parameters.discretized_domain, training_data):
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
