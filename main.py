import math

import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from finite_difference import FiniteDifference
from neural_network import NeuralNetwork

left_domain_boundary = 0.5
right_domain_boundary = 1
nr_of_grid_points = 50

def activation_function(x):
    return x

# Initialize the list of points where the solution of the ODE will be approximated
grid_points = numpy.linspace(
    left_domain_boundary,
    right_domain_boundary,
    nr_of_grid_points
)

# Initialize the finite difference solver, to generate training data
ode_solver = FiniteDifference(grid_points, 1, 1, math.cos, math.sin)

# Generate training data
training = ode_solver.solve()

# Initialize neural network using the sigmoid activation function and 2 hidden layers with 5 nodes each
net = NeuralNetwork(torch.relu_, [20, 50, 20])

# Use Mean Squared Error as the loss function
loss_fn = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network
for t in range(3000):
    # One training means comparing the output of the neural network for every grid point with the solution
    # computed by the finite difference solver and updating the weights accordingly
    for i in range(len(grid_points)):
        # Initialize neural network input
        net_input = torch.tensor(
            grid_points[i], dtype=torch.float32).unsqueeze(dim=0)

        # Initialize predicted output
        pred_output = torch.tensor(
            training[i], dtype=torch.float32).unsqueeze(dim=0)

        # Compute the neural network output
        net_output = net(net_input)

        # Compute the loss using Mean Squared Error loss function
        loss = loss_fn(pred_output, net_output)

        # Reset the gradient buffers
        net.zero_grad()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()


net_solutions = []

# Use the trained neural network to compute the solution of the ODE at each grid point
for i in range(len(grid_points)):
    net_input = torch.tensor(
            grid_points[i], dtype=torch.float32).unsqueeze(dim=0)

    net_solutions.append(net(net_input).item())

# Plot the solutions
plot_neural_net = plt.plot(grid_points, net_solutions,
                           label="neural network solution")

plot_finite_difference = plt.plot(
    grid_points, training, label="finite difference solution")

plt.legend()
plt.show()
