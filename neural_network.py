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
import torch.nn.functional as F

class Neural_Network(nn.Module):
    def __init__(self, hiddenLayers):
        super(Neural_Network, self).__init__()
        
        # Set input and output size
        self.inputSize = 1
        self.outputSize = 1

        # Initialize weights
        self.weights = list()

        # Set random weights for the input layer
        self.weights.append(torch.randn(self.inputSize, hiddenLayers[0]))

        # Set random weights for each hidden layer
        if (isinstance(hiddenLayers, list)):
            for i in range(1, len(hiddenLayers)):
                self.weights.append(torch.randn(hiddenLayers[i - 1], hiddenLayers[i]))

        # Set random weights for the output layer
        self.weights.append(torch.randn(hiddenLayers(len(hiddenLayers) - 1), self.outputSize))

        # Set random biases
        self.biases = torch.randn(len(hiddenLayers) + 1)

    # TODO
    def forward(self, X):
        return

    # TODO
    def backpropagation(self, ):
        return

    # Forward Rectified Linear Unit activation function
    def ReLU(self, s):
        return max(0, s)

    # Backward Rectified Linear Unit activation function
    def backpropReLU(self, s):
        return max(0, 1)

    # Forward sigmoid activation function
    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    # Backward sigmoid activation function
    def backpropSigmoid(self, s):
        return s * (1 - s)

    