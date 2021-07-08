import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Model


class NeuralNetwork(Model):
    """A NeuralNetwork class implements a standard neural network architecture.

    """

    def __init__(self, n_input=784, n_hidden=384, n_output=10, lr=0.001, init_weights=None, device='cpu'):
        """Initialization method.

        Args:
            n_input (int): Number of input units.
            n_hidden (int): Number of hidden units.
            n_output (int): Number of output units.
            lr (float): Learning rate.
            init_weights (tuple): Tuple holding the minimum and maximum values for weights initialization.
            device (str): Device that model should be trained on, e.g., `cpu` or `cuda`.

        """

        # Override its parent class
        super(NeuralNetwork, self).__init__(init_weights, device)

        # Number of inputs
        self.n_input = n_input

	    # Fully-connected layers
        # (n_input, n_output)
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

        # Dropout layer
        # (probability)
        self.drop = nn.Dropout(0.25)

        # Compiles the network
        self._compile(lr)

    def forward(self, x):
        """Performs the forward pass.

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            Output tensor.

        """

        # Arranges the input as a continuous amount of units
        x = x.view(-1, self.n_input)
        
        # Fully-connected block
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)

        return x
