# Wir haben zum Beispiel die Differentialgleichung  -u''(t) + a(t)*u(t) = phi(t).
# Abhängig von a und phi erhalten wir verschiedene Lösungen, d.h wir haben a und phi als Input
# und vielleicht Randwerte/ Anfangswerte und errechnen daraus unser u(t).
# Wir haben damit unser Input-Output-Paar.
#
# Wir möchten jetzt direkt eine Abbildung erlernen die uns den Input auf den Output abbildet.
# Die Gleichung muss dann zunächst mit dem Differenzenverfahren gelöst werden, um Trainingsdaten zu erhalten.
#
# Training des neuronalen Netzes:
# Inputs: a (Funktion), phi (Funktion), alpha_0, alpha_1,
# Output: u (Neuronales Netz, Funktion)
#
# Neuronales Netz:
# Input: t
# Output: u(t)

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, activation_function, hidden_layers):
        # Initialization
        super(NeuralNetwork, self).__init__()
        self.activation_function = activation_function

        # Set input and output size
        self.input_size = 1
        self.output_size = 1

        if (isinstance(hidden_layers, list)):
            # Initialize hidden layers
            self.hidden = nn.ModuleList()

            # Set linear transformation for input layer to first hidden layer
            self.hidden.append(nn.Linear(self.input_size, hidden_layers[0]))

            # Set linear transformation for the remaining hidden layers
            for i in range(0, len(hidden_layers) - 1):
                self.hidden.append(
                    nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

            # Set linear transformation for last hidden layer to output layer
            self.hidden.append(
                nn.Linear(hidden_layers[len(hidden_layers) - 1], self.output_size))
        else:
            raise ValueError(
                "Expected list containing the number of nodes in the hidden layers.")

    def forward(self, x):
        for i in range(0, len(self.hidden)):
            # Perform linear transformation
            x = self.hidden[i](x)

            # Apply forward activation function
            x = self.activation_function(x)

        return x
